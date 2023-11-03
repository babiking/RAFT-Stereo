import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import (
    CorrBlock1D,
    PytorchAlternateCorrBlock1D,
    CorrBlockFast1D,
    AlternateCorrBlock,
)
from core.utils.utils import coords_grid, upflow8


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTStereo(nn.Module):
    def __init__(
        self,
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=True,
        n_gru_layers=3,
        mixed_precision=True,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.context_dims = copy.deepcopy(hidden_dims)
        self.corr_implementation = corr_implementation
        self.shared_backbone = shared_backbone
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.n_downsample = n_downsample
        self.context_norm = context_norm
        self.slow_fast_gru = slow_fast_gru
        self.n_gru_layers = n_gru_layers
        self.mixed_precision = mixed_precision

        self.cnet = MultiBasicEncoder(
            output_dim=[self.hidden_dims, self.context_dims],
            norm_fn=self.context_norm,
            downsample=self.n_downsample,
        )
        self.update_block = BasicMultiUpdateBlock(
            self.hidden_dims,
            self.corr_levels,
            self.corr_radius,
            self.n_gru_layers,
            self.n_downsample,
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2
                )
                for i in range(self.n_gru_layers)
            ]
        )

        if self.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", downsample=self.n_downsample
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        # coords0: 1 x 2 x 120 x 160
        coords0 = coords_grid(N, H, W).to(img.device)
        # coords1: 1 x 2 x 120 x 160
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        # flow: 1 x 2 x 120 x 160
        # mask: 1 x 144 x 120 x 160
        N, D, H, W = flow.shape
        factor = 2**self.n_downsample
        # mask: 1 x 1 x 9 x 4 x 4 x 120 x 160
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        # mask: 1 x 1 x 9 x 4 x 4 x 120 x 160
        mask = torch.softmax(mask, dim=2)

        # up_flow: 1 x (2 * 9) x (120 * 160)
        up_flow = F.unfold(factor * flow, [1, 9], padding=[0, 4])
        # up_flow: 1 x 2 x 9 x 1 x 1 x 120 x 160
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        # up_flow: 1 x 2 x 4 x 4 x 120 x 160
        up_flow = torch.sum(mask * up_flow, dim=2)
        # up_flow: 1 x 2 x 120 x 4 x 160 x 4
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # up_flow: 1 x 2 x 480 x 640
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """Estimate optical flow between pair of frames"""

        # image: 1 x 3 x 480 x 640, format NCHW
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.mixed_precision):
            if self.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=self.n_gru_layers,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                # cnet_list:
                #   [1] outputs08
                #       a. hidden_dims:  1 x 128 x 120 x 160
                #       b. context_dims: 1 x 128 x 120 x 160
                #   [2] outputs16
                #       a. hidden_dims:  1 x 128 x 60 x 80
                #       b. context_dims: 1 x 128 x 60 x 80
                #   [3] outputs32
                #       a. hidden_dims:  1 x 128 x 30 x 40
                #       b. context_dims: 1 x 128 x 30 x 40
                cnet_list = self.cnet(image1, num_layers=self.n_gru_layers)
                # fnet:
                #   [1] fmap1: 1 x 128 x 120 x 160
                #   [2] fmap2: 1 x 128 x 120 x 160
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            # inp_list i.e. context_dims:
            #   [1] outputs08
            #       [1.1] 1 x 128 x 120 x 160
            #       [1.2] 1 x 128 x 120 x 160
            #       [1.3] 1 x 128 x 120 x 160
            #   [2] outputs16
            #       [2.1] 1 x 128 x 60 x 80
            #       [2.2] 1 x 128 x 60 x 80
            #       [2.3] 1 x 128 x 60 x 80
            #   [3] outputs32
            #       [3.1] 1 x 128 x 30 x 40
            #       [3.2] 1 x 128 x 30 x 40
            #       [3.3] 1 x 128 x 30 x 40
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.corr_implementation == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.corr_implementation == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(
            fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels
        )
        
        # coords0: 1 x 2 x 120 x 160
        # coords1: 1 x 2 x 120 x 160
        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            # corr: 1 x 36 x 120 x 160
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                if self.n_gru_layers == 3 and self.slow_fast_gru:  # Update low-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )
                if (
                    self.n_gru_layers >= 2 and self.slow_fast_gru
                ):  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.n_gru_layers == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                # net_list, i.e. hidden:
                #   [1] 1 x 128 x 120 x 160
                #   [2] 1 x 128 x 60 x 80
                #   [3] 1 x 128 x 30 x 40
                # up_mask: 1 x 36 x 120 x 160
                # delta_flow: 1 x 2 x 120 x 160
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.n_gru_layers == 3,
                    iter16=self.n_gru_layers >= 2,
                )

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                # coords1 - coords0: 1 x 2 x 120 x 160
                # up_mask: 1 x 36 x 120 x 160
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
