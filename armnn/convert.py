import os
import torch
import onnx
from deprecated import deprecated
from google.protobuf.json_format import MessageToDict
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
    mixed_precision=False,
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


def change_onnx_input_dtype(onnx_model_file):
    # reference: https://stackoverflow.com/questions/56734576/find-input-shape-from-onnx-file
    model = onnx.load(onnx_model_file)
    for msg_in in model.graph.input:
        dict_in = MessageToDict(msg_in)
        """
        onnx datatype:
            elem_type: 1 --> float32
            elem_type: 2 --> uint8
            elem_type: 3 --> int8
            elem_type: 4 --> uint16
            elem_type: 5 --> int16
            elem_type: 6 --> int32
            elem_type: 7 --> int64
            elem_type: 8 --> string
            elem_type: 9 --> boolean
            elem_type: 10 --> float16
            elem_type: 11 --> float64
            elem_type: 12 --> uint32
            elem_type: 14 --> uint64
            elem_type: 15 --> complex128
            elem_type: 16 --> bfloat16
        """
        if msg_in.name in ["left", "right"]:
            msg_in.type.tensor_type.elem_type = 1
    onnx.save(model, onnx_model_file)


@deprecated(version="v1.0", reason="onnx-tensorflow NOT support GridSampler op!")
def convert_onnx_to_tf(onnx_model_file, tf_model_file):
    import onnx_tf
    onnx_model = onnx.load(onnx_model_file)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(
        os.path.join(
            os.path.dirname(tf_model_file),
            os.path.splitext(os.path.basename(tf_model_file))[0],
        )
    )