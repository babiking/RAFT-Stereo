from __future__ import print_function, division
import os
import sys
import json
import logging
import numpy as np
import torch
import cv2 as cv
from tools.pfm_file_io import write_pfm_file
from tools.colorize import colorize_2d_matrix
from core.raft_stereo import RAFTStereo, autocast
from core.utils.utils import InputPadder
import gflags

gflags.DEFINE_string(
    "exp_config_json", "configure/exp_config.json", "experiment configure json file"
)
gflags.DEFINE_string(
    "model_chkpt_file",
    "experiments/BASE_FOLD_1_9/checkpoints/BASE_FOLD_1_9-epoch-100000.pth.gz",
    "model checkpont file",
)
gflags.DEFINE_string("left", "armnn/test/data/300cm_left_Img.bmp", "left image file")
gflags.DEFINE_string(
    "right", "armnn/test/data/300cm_right_Img.bmp", "right image file"
)
gflags.DEFINE_float("fx", 313.422, "focal length in pixel")
gflags.DEFINE_float("baseline", 3.53297, "baseline in cm")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.exp_config_json, "r"))

    model = torch.nn.DataParallel(RAFTStereo(**exp_config["model"])).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.cuda()
    model.eval()

    assert FLAGS.model_chkpt_file.endswith(".pth") or FLAGS.model_chkpt_file.endswith(
        ".pth.gz"
    )
    logging.info(f"Loading checkpoint: {FLAGS.model_chkpt_file}...")
    checkpoint = torch.load(FLAGS.model_chkpt_file)
    model.load_state_dict(checkpoint, strict=True)
    logging.info(f"Done loading checkpoint.")

    print(
        f"The model has {format(count_parameters(model)/1e6, '.4f')}M learnable parameters."
    )

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = exp_config["model"]["corr_implementation"].endswith("_cuda")

    valid_iters = exp_config["test"]["validate_num_of_updates"]

    image1 = cv.imread(FLAGS.left, cv.IMREAD_UNCHANGED)
    image2 = cv.imread(FLAGS.right, cv.IMREAD_UNCHANGED)

    image1 = image1[:, :, ::-1]
    image2 = image2[:, :, ::-1]

    image1 = torch.from_numpy(image1.copy()).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2.copy()).permute(2, 0, 1).float()

    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with autocast(enabled=use_mixed_precision):
        _, flow_pr = model(image1, image2, iters=valid_iters, test_mode=True)
    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

    flow_pr = -1.0 * np.squeeze(flow_pr.cpu().detach().numpy(), axis=0)

    depth_pr = FLAGS.baseline * FLAGS.fx / (flow_pr + 1e-12)
    depth_pr[np.where(flow_pr < 1e-9)] = 0.0

    depth_color = colorize_2d_matrix(depth_pr, min_val=10.0, max_val=320.0)

    data_path = os.path.dirname(FLAGS.left)
    data_name = os.path.splitext(os.path.basename(FLAGS.left))[0]

    depth_pfm_file = os.path.join(data_path, f"{data_name}.pfm")
    write_pfm_file(depth_pfm_file, np.flipud(depth_pr), 1.0)

    depth_color_file = os.path.join(data_path, f"{data_name}_colorize.png")
    cv.imwrite(depth_color_file, depth_color[:, :, ::-1])


if __name__ == "__main__":
    main()
