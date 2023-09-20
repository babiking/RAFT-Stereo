import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )

    def forward(self, h, cz, cr, cq, *x_list):
        # x: 1 x 128 x 30 x 40
        x = torch.cat(x_list, dim=1)
        # hx: 1 x 256 x 30 x 40
        hx = torch.cat([h, x], dim=1)

        # z: 1 x 128 x 30 x 40
        z = torch.sigmoid(self.convz(hx) + cz)
        # r: 1 x 128 x 30 x 40
        r = torch.sigmoid(self.convr(hx) + cr)
        # q: 1 x 128 x 30 x 40
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        # h: 1 x 128 x 30 x 40
        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        cor_planes = self.corr_levels * (2 * self.corr_radius + 1) # cor_planes = 36

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        # cor: 1 x 64 x 120 x 160
        cor = F.relu(self.convc1(corr))
        # cor: 1 x 64 x 120 x 160
        cor = F.relu(self.convc2(cor))
        # flo: 1 x 64 x 120 x 160
        flo = F.relu(self.convf1(flow))
        # flo: 1 x 64 x 120 x 160
        flo = F.relu(self.convf2(flo))

        # cor_flo: 1 x 128 x 120 x 160
        cor_flo = torch.cat([cor, flo], dim=1)
        # out: 1 x 126 x 120 x 160
        out = F.relu(self.conv(cor_flo))
        # out: 1 x 128 x 120 x 160
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {"mode": "bilinear", "align_corners": True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(
        self, hidden_dims, corr_levels, corr_radius, n_gru_layers, n_downsample
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.n_gru_layers = n_gru_layers
        self.n_downsample = n_downsample

        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(
            hidden_dims[2], # hidden_dim = 128
            encoder_output_dim + hidden_dims[1] * (self.n_gru_layers > 1), # input_dim = 256
        )
        self.gru16 = ConvGRU(
            hidden_dims[1], # hidden_dim = 128 
            hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2], # input_dim = 256
        )
        self.gru32 = ConvGRU(
            hidden_dims[0], # hidden_dim = 128 
            hidden_dims[1], # input_dim = 128
        )
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2) * 9, 1, padding=0),
        )

    def forward(
        self,
        net,
        inp,
        corr=None,
        flow=None,
        iter08=True,
        iter16=True,
        iter32=True,
        update=True,
    ):
        # net, i.e. hidden:
        #   [1] 1 x 128 x 120 x 160
        #   [2] 1 x 128 x 60 x 80
        #   [3] 1 x 128 x 30 x 40
        # inp, i.e. context:
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
        if iter32:
            # gru32: hidden_dim = 128, input_dim = 128, 1 x 128 x 30 x 40
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                # gru16: hidden_dim = 128, input_dim = 256, 1 x 128 x 60 x 80
                net[1] = self.gru16(
                    net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1])
                )
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            # corr: 1 x 36 x 120 x 160
            # flow: 1 x 2 x 120 x 160
            # motion_features: 1 x 128 x 120 x 160
            motion_features = self.encoder(flow, corr)
            if self.n_gru_layers > 1:
                # gru08: hidden_dim = 128, input_dim = 256, 1 x 128 x 120 x 160
                net[0] = self.gru08(
                    net[0], *(inp[0]), motion_features, interp(net[1], net[0])
                )
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            # net, i.e. hidden:
            #   [1] 1 x 128 x 120 x 160
            #   [2] 1 x 128 x 60 x 80
            #   [3] 1 x 128 x 30 x 40
            return net

        # delta_flow: 1 x 2 x 120 x 160
        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        # mask: 1 x 9 * 4^(2) x 120 x 160, i.e. 1 x 144 x 120 x 160
        mask = 0.25 * self.mask(net[0])
        return net, mask, delta_flow
