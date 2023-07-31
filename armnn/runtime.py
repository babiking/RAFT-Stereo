import cv2 as cv
import numpy as np
import onnxruntime


def run_onnx_model_inference(onnx_model_file, l_img_file, r_img_file):
    session = onnxruntime.InferenceSession(onnx_model_file)

    l_img = cv.imread(l_img_file, cv.IMREAD_UNCHANGED)[np.newaxis, :, :, ::-1].astype(
        np.float32
    )
    l_img = np.transpose(l_img, [0, 3, 1, 2])

    r_img = cv.imread(r_img_file, cv.IMREAD_UNCHANGED)[np.newaxis, :, :, ::-1].astype(
        np.float32
    )
    r_img = np.transpose(r_img, [0, 3, 1, 2])

    _, l_disp = session.run(
        None,
        {"left": l_img, "right": r_img},
    )
    return l_disp