import pytest

import os
import glob
import numpy as np
from tqdm import tqdm
from armnn.runtime import run_torch_model_inference, run_onnx_model_inference


def test_onnx_consistency():
    work_path = os.path.dirname(__file__)

    data_idxes = [
        os.path.splitext(os.path.basename(x))[0].split("_")[-1]
        for x in glob.glob(os.path.join(work_path, "data/left_*.png"))
    ]

    pt_model_file = "models/raftstereo-middlebury.pth"
    onnx_model_file = "models/raftstereo-middlebury.onnx"

    pbar = tqdm(total=len(data_idxes))
    for data_idx in data_idxes:
        l_img_file = os.path.join(work_path, f"data/left_{data_idx}.png")
        r_img_file = os.path.join(work_path, f"data/right_{data_idx}.png")

        l_disp_torch = run_torch_model_inference(
            pt_model_file,
            l_img_file,
            r_img_file,
            device="cuda",
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
            n_flow_updates=32,
            division=32,
        )

        l_disp_onnx = run_onnx_model_inference(onnx_model_file, l_img_file, r_img_file)

        l_disp_mean_diff = np.mean(np.abs(l_disp_onnx - l_disp_torch))
        
        assert l_disp_mean_diff < 0.05, f"{data_idx} stereo image pytorch and onnx mismatch!"

        pbar.update(1)
        pbar.set_description(f"{data_idx} stereo image pytorch and onnx match.")


if __name__ == "__main__":
    test_onnx_consistency()
