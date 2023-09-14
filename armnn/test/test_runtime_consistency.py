import pytest

import os
import glob
import torch
import onnx2tf
import numpy as np
from matplotlib import pyplot as plt
from armnn.convert import convert_torch_to_onnx
from armnn.runtime import read_image_file, run_torch_model_inference, run_onnx_model_inference, run_tflite_model_inference
from tools.logger import create_color_logger


def test_onnx2tf_consistency():
    logger = create_color_logger(name="test-onnx2tf-consistency")

    work_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))

    kwargs = {
        "width": 640,
        "height": 480,
        "hidden_dims": [128, 128, 128],
        "corr_implementation": "reg",
        "shared_backbone": False,
        "corr_levels": 4,
        "corr_radius": 4,
        "n_downsample": 2,
        "context_norm": "batch",
        "slow_fast_gru": True,
        "n_gru_layers": 3,
        "mixed_precision": True,
        "n_flow_updates": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    model_name = "raftstereo-middlebury"
    torch_model_file = os.path.join(work_path, f"models/{model_name}.pth")
    logger.info(f"found torch model file: {torch_model_file}.")

    onnx_model_file = os.path.join(work_path, f"models/{model_name}.onnx")
    if not os.path.exists(onnx_model_file) \
        or os.path.getsize(onnx_model_file) <= 0:
        logger.info(f"start torch export to onnx model conversion....")
        convert_torch_to_onnx(\
            torch_model_file, onnx_model_file, batch_size=1, **kwargs)
    logger.info(f"converted onnx model file: {onnx_model_file}.")

    onnx2tf_save_name = "saved_model"
    tflite_model_path = os.path.join(work_path, f"models/{onnx2tf_save_name}")
    tflite_model_file = \
        os.path.join(tflite_model_path, f"{model_name}_float32.tflite")
    param_replace_json = os.path.join(
        work_path, "armnn/replacement/param_replacement.json")
    if not os.path.exists(tflite_model_file) \
        or os.path.getsize(tflite_model_file) <= 0:
        logger.info(
            f"start onnx2tf model conversion, onnx2tf-{onnx2tf.__version__}..."
        )
        onnx2tf.convert(input_onnx_file_path=onnx_model_file,
                        output_folder_path=tflite_model_path,
                        param_replacement_file=param_replace_json,
                        disable_strict_mode=False)
    logger.info(f"converted tflite model file: {tflite_model_file}.")

    debug_path = os.path.join(work_path, "armnn/test/debug")
    os.makedirs(debug_path, exist_ok=True)

    kwargs.pop("width")
    kwargs.pop("height")
    data_idxes = [
        os.path.splitext(os.path.basename(x))[0].split("_")[-1]
        for x in glob.glob(
            os.path.join(work_path, "armnn/test/data/left_*.png"))
    ]
    for data_idx in data_idxes:
        l_img_file = os.path.join(work_path,
                                  f"armnn/test/data/left_{data_idx}.png")
        r_img_file = os.path.join(work_path,
                                  f"armnn/test/data/right_{data_idx}.png")

        l_img = read_image_file(l_img_file)
        r_img = read_image_file(r_img_file)

        l_disp_torch = run_torch_model_inference(l_img,
                                                 r_img,
                                                 torch_model_file,
                                                 division=32,
                                                 **kwargs)
        plt.imsave(os.path.join(debug_path, f"letf_{data_idx}_torch.png"),
                   np.squeeze(l_disp_torch),
                   cmap="jet")
        logger.info(f"{data_idx} torch inference completed.")

        l_disp_onnx = run_onnx_model_inference(l_img, r_img, onnx_model_file)
        plt.imsave(os.path.join(debug_path, f"letf_{data_idx}_onnx.png"),
                   np.squeeze(l_disp_onnx),
                   cmap="jet")
        l_disp_err = np.mean(np.abs(l_disp_torch - l_disp_onnx))
        assert l_disp_err < 0.05, f"{data_idx} stereo image pytorch and [onnx] mismatch!"
        logger.info(
            f"{data_idx} onnx inference completed with average disparity error {l_disp_err:.4f}."
        )

        l_disp_tflite = \
            run_tflite_model_inference(l_img, r_img, tflite_model_file)
        plt.imsave(os.path.join(debug_path, f"letf_{data_idx}_tflite.png"),
                   np.squeeze(l_disp_tflite),
                   cmap="jet")
        l_disp_err = np.mean(np.abs(l_disp_torch - l_disp_tflite))
        assert l_disp_err < 0.5, f"{data_idx} stereo image pytorch and [tflite] mismatch!"
        logger.info(
            f"{data_idx} tflite inference completed with average disparity error {l_disp_err:.4f}."
        )


if __name__ == "__main__":
    test_onnx2tf_consistency()
