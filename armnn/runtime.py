import torch
import cv2 as cv
import numpy as np
import onnxruntime
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


def read_image_file(img_file):
    img = cv.imread(img_file, cv.IMREAD_UNCHANGED)

    if (len(img.shape) < 3) or (len(img.shape) == 3 and img.shape[-1] == 1):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # color format BGR to RGB
    img = img[:, :, ::-1]

    img = img[np.newaxis, :, :, :]

    # torch feature map dimension format: NCHW
    img = np.transpose(img, [0, 3, 1, 2])
    return img


def run_torch_model_inference(
    l_img,
    r_img,
    torch_model_file,
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
):
    torch_l_img = torch.tensor(l_img.copy(), dtype=torch.float32).to(device)
    torch_r_img = torch.tensor(r_img.copy(), dtype=torch.float32).to(device)

    padder = InputPadder(torch_l_img.shape, divis_by=division)
    torch_l_img, torch_r_img = padder.pad(torch_l_img, torch_r_img)

    model = torch.nn.DataParallel(
        RAFTStereo(
            hidden_dims=hidden_dims,
            corr_implementation=corr_implementation,
            shared_backbone=shared_backbone,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            n_downsample=n_downsample,
            context_norm=context_norm,
            slow_fast_gru=slow_fast_gru,
            n_gru_layers=n_gru_layers,
            mixed_precision=mixed_precision,
        ),
        device_ids=[0],
    )
    model.load_state_dict(
        torch.load(torch_model_file, map_location=torch.device(device)))

    model = model.module
    model.to(device)
    model.eval()

    _, torch_l_disp = model(torch_l_img,
                            torch_r_img,
                            iters=n_flow_updates,
                            test_mode=True)
    torch_l_disp = padder.unpad(torch_l_disp).squeeze()
    torch_l_disp *= -1.0
    return torch_l_disp.cpu().detach().numpy().squeeze()


def run_onnx_model_inference(l_img, r_img, onnx_model_file):
    session = onnxruntime.InferenceSession(onnx_model_file)

    _, onnx_l_disp = session.run(
        None,
        {
            "left": l_img.astype(np.float32),
            "right": r_img.astype(np.float32)
        },
    )
    onnx_l_disp *= -1.0
    return np.squeeze(onnx_l_disp)


def run_tflite_model_inference(l_img, r_img, tflite_model_file):
    tflite_l_img = np.transpose(l_img, [0, 2, 3, 1])
    tflite_r_img = np.transpose(r_img, [0, 2, 3, 1])

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"],
        tf.convert_to_tensor(tflite_l_img, dtype=tf.float32))
    interpreter.set_tensor(
        input_details[1]["index"],
        tf.convert_to_tensor(tflite_r_img, dtype=tf.float32))

    interpreter.invoke()

    tflite_l_disp = interpreter.get_tensor(output_details[1]["index"])
    tflite_l_disp *= -1.0
    return tflite_l_disp


def run_armnn_delegate_model_inference(
    l_img,
    r_img,
    armnn_delegate_file,
    tflite_model_file,
):
    armnn_delegate = tflite.load_delegate(library=armnn_delegate_file,
                                          options={
                                              "backends":
                                              "CpuAcc,GpuAcc,CpuRef",
                                              "logging-severity": "info"
                                          })

    # Delegates/Executes all operations supported by Arm NN to/with Arm NN
    interpreter = tflite.Interpreter(model_path=tflite_model_file,
                                     experimental_delegates=[armnn_delegate])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    armnn_l_img = np.transpose(l_img, [0, 2, 3, 1])
    armnn_r_img = np.transpose(r_img, [0, 2, 3, 1])

    interpreter.set_tensor(input_details[0]["index"],
                           armnn_l_img.astype(np.float32))
    interpreter.set_tensor(input_details[1]["index"],
                           armnn_r_img.astype(np.float32))

    interpreter.invoke()

    armnn_l_disp = interpreter.get_tensor(output_details[1]["index"])
    armnn_l_disp *= -1.0
    return armnn_l_disp