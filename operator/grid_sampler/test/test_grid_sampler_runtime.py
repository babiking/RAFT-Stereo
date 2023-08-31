import pytest

import os
import time
from functools import partial
import torch
import torch.nn.functional as F
import onnxruntime
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tools.logger import create_color_logger


def run_torch_grid_sampler(input, grid):
    torch_output = F.grid_sample(
        input=torch.tensor(input, dtype=torch.float32),
        grid=torch.tensor(grid, dtype=torch.float32),
        padding_mode="zeros",
        mode="bilinear",
        align_corners=True,
    )
    torch_output = torch_output.cpu().detach().numpy()
    return torch_output


def run_onnx_grid_sampler(input, grid, onnx_model_file):
    session = onnxruntime.InferenceSession(onnx_model_file)

    onnx_output = session.run(None, {"input": input, "grid": grid})
    return onnx_output[0]


def run_tflite_grid_sampler(input, grid, tflite_model_file):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"], tf.convert_to_tensor(input, dtype=tf.float32)
    )
    interpreter.set_tensor(
        input_details[1]["index"], tf.convert_to_tensor(grid, dtype=tf.float32)
    )

    interpreter.invoke()

    tflite_output = interpreter.get_tensor(output_details[0]["index"])
    # TFLite dimension: NHWC -> NCHW
    tflite_output = np.transpose(tflite_output, [0, 3, 1, 2])
    return tflite_output


def run_armnn_delegate_grid_sampler(input, grid, tflite_model_file, armnn_lib_file):
    armnn_delegate = tflite.load_delegate(
        library=armnn_lib_file,
        options={"backends": "CpuAcc,GpuAcc,CpuRef", "logging-severity": "info"},
    )

    interpreter = tflite.Interpreter(
        model_path=tflite_model_file, experimental_delegates=[armnn_delegate]
    )
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"], tf.convert_to_tensor(input, dtype=tf.float32)
    )
    interpreter.set_tensor(
        input_details[1]["index"], tf.convert_to_tensor(grid, dtype=tf.float32)
    )

    interpreter.invoke()

    armnn_output = interpreter.get_tensor(output_details[0]["index"])
    # TFLite dimension: NHWC -> NCHW
    armnn_output = np.transpose(armnn_output, [0, 3, 1, 2])
    return armnn_output


def main():
    logger = create_color_logger(name="test-grid-sampler-runtime")

    # pytorch 4D feature map dimension: NCHW
    n = 32
    c = 16
    h = 36
    w = 64

    input = np.random.randn(n, c, h, w).astype(np.float32)
    grid = np.random.random(size=[n, h, w, 2]).astype(np.float32)
    grid = 2.0 * grid - 1.0

    grid_sampler_funcs = {
        "PYTORCH": run_torch_grid_sampler,
        "ONNX": partial(
            run_onnx_grid_sampler,
            onnx_model_file="operator/grid_sampler/grid_sampler_models/grid_sample_reproduction.onnx",
        ),
        "TFLITE": partial(
            run_tflite_grid_sampler,
            tflite_model_file="operator/grid_sampler/grid_sampler_models/grid_sample_reproduction.tf/grid_sample_reproduction_float32.tflite",
        ),
        "ARMNN": partial(
            run_armnn_delegate_grid_sampler,
            tflite_model_file="operator/grid_sampler/grid_sampler_models/grid_sample_reproduction.tf/grid_sample_reproduction_float32.tflite",
            armnn_lib_file="thirdparty/armnn/libarmnnDelegate.so",
        ),
    }

    grid_sampler_outputs = {}
    for framework, grid_sampler_func in grid_sampler_funcs.items():
        start = time.time()
        grid_sampler_outputs[framework] = grid_sampler_func(input, grid)
        end = time.time()
        logger.info(f"{framework} elapsed time: {float(end - start):.4f} seconds.")

    groundtruth = "PYTORCH"
    for framework in grid_sampler_outputs.keys():
        if framework == groundtruth:
            continue

        assert np.allclose(
            grid_sampler_outputs[framework],
            grid_sampler_outputs[groundtruth],
            atol=1e-6,
            rtol=1e-6,
        ), f"{framework} runtime consistency check failed!"


if __name__ == "__main__":
    main()
