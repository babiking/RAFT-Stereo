import torch
import cv2 as cv
import numpy as np
import onnxruntime
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


def read_image_file(img_file):
    # torch feature map dimension format: NCHW
    img = cv.imread(img_file, cv.IMREAD_UNCHANGED)[np.newaxis, :, :, ::-1].astype(
        np.float32
    )
    img = np.transpose(img, [0, 3, 1, 2])
    return img


def run_torch_model_inference(
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
):
    l_img = read_image_file(l_img_file)
    l_img = torch.tensor(l_img, dtype=torch.float32).to(device)

    r_img = read_image_file(r_img_file)
    r_img = torch.tensor(r_img, dtype=torch.float32).to(device)

    padder = InputPadder(l_img.shape, divis_by=division)
    l_img, r_img = padder.pad(l_img, r_img)

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
    model.load_state_dict(torch.load(pt_model_file))

    model = model.module
    model.to(device)
    model.eval()

    _, l_disp = model(l_img, r_img, iters=n_flow_updates, test_mode=True)
    l_disp = padder.unpad(l_disp).squeeze()
    l_disp *= -1.0

    return l_disp.cpu().detach().numpy().squeeze()


def run_onnx_model_inference(onnx_model_file, l_img_file, r_img_file):
    session = onnxruntime.InferenceSession(onnx_model_file)

    l_img = read_image_file(l_img_file)
    r_img = read_image_file(r_img_file)

    _, l_disp = session.run(
        None,
        {"left": l_img, "right": r_img},
    )
    l_disp *= -1.0

    return np.squeeze(l_disp)
