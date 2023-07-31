import os
import sys
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

import gflags

gflags.DEFINE_string("device", "cuda", "pytorch inference device, cpu | cuda")
gflags.DEFINE_string(
    "checkpoint", "models/raftstereo-middlebury.pth", "pytorch checkpoint .pt file"
)
gflags.DEFINE_string("left", "armnn/test/data/left_0000.png", "left image")
gflags.DEFINE_string("right", "armnn/test/data/right_0000.png", "right image")
gflags.DEFINE_string("save_path", "armnn/test/disparity", "save path")
gflags.DEFINE_integer(
    "n_flow_updates", 32, "number of flow-field updates during forward pass"
)
gflags.DEFINE_list(
    "hidden_dims", [128, 128, 128], "hidden state dimension of backbone network"
)
gflags.DEFINE_string(
    "corr_implementation",
    "reg",
    "correlation volume implementation method, reg | alt | reg_cuda | alt_cuda",
)
gflags.DEFINE_boolean("shared_backbone", False, "if use shared backbone")
gflags.DEFINE_integer("corr_levels", 4, "number of levels in the correlation pyramid")
gflags.DEFINE_integer("corr_radius", 4, "width of the correlation pyramid")
gflags.DEFINE_integer("n_downsample", 2, "resolution of the disparity field (1/2^K)")
gflags.DEFINE_string(
    "context_norm",
    "batch",
    "normalization of context encoder, group | batch | instance | none",
)
gflags.DEFINE_string(
    "slow_fast_gru", True, "iterate the low-resolution GRUs more frequently"
)
gflags.DEFINE_integer("n_gru_layers", 3, "number of hidden GRU levels")
gflags.DEFINE_boolean("mixed_precision", True, "if use mixed precision")


def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def demo():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    model = torch.nn.DataParallel(
        RAFTStereo(
            hidden_dims=FLAGS.hidden_dims,
            corr_implementation=FLAGS.corr_implementation,
            shared_backbone=FLAGS.shared_backbone,
            corr_levels=FLAGS.corr_levels,
            corr_radius=FLAGS.corr_radius,
            n_downsample=FLAGS.n_downsample,
            context_norm=FLAGS.context_norm,
            slow_fast_gru=FLAGS.slow_fast_gru,
            n_gru_layers=FLAGS.n_gru_layers,
            mixed_precision=FLAGS.mixed_precision,
        ),
        device_ids=[0],
    )
    model.load_state_dict(torch.load(FLAGS.checkpoint))

    model = model.module
    model.to(FLAGS.device)
    model.eval()

    output_directory = Path(FLAGS.save_path)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(FLAGS.left, recursive=True))
        right_images = sorted(glob.glob(FLAGS.right, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1, FLAGS.device)
            image2 = load_image(imfile2, FLAGS.device)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(
                image1, image2, iters=FLAGS.n_flow_updates, test_mode=True
            )
            flow_up = padder.unpad(flow_up).squeeze()

            file_stem = os.path.splitext(os.path.basename(imfile1))[0]
            plt.imsave(
                output_directory / f"{file_stem}.png",
                -flow_up.cpu().numpy().squeeze(),
                cmap="jet",
            )


if __name__ == "__main__":
    demo()
