import torch
from core.raft_stereo import RAFTStereo


def convert_torch_to_onnx(
    pt_model_file,
    onnx_model_file,
    device="cuda",
    batch_size=1,
    width=640,
    height=480,
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
):
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

    # pytorch input format: NCHW
    sample_left = torch.rand((batch_size, 3, height, width), dtype=torch.float32)
    sample_left = sample_left.to(device)

    sample_right = torch.rand((batch_size, 3, height, width), dtype=torch.float32)
    sample_right = sample_right.to(device)

    sample_n_flow_updates = torch.tensor(n_flow_updates, dtype=torch.int)

    torch.onnx.export(
        model,
        (sample_left, sample_right, sample_n_flow_updates, None, True),
        onnx_model_file,
        verbose=False,
        input_names=["left", "right", "n_flow_updates", "flow_init", "test_mode"],
        output_names=["disparity"],
        opset_version=16,
    )
