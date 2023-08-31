import os
import torch
import torch.nn.functional as F
import onnx2tf


class GridSamplerOperator(torch.nn.Module):
    def forward(self, input: torch.Tensor, grid: torch.Tensor):
        return F.grid_sample(
            input=input,
            grid=grid,
            padding_mode="zeros",
            mode="bilinear",
            align_corners=True,
        )


def main():
    grid_sampler = GridSamplerOperator()

    # pytorch 4D feature map dimension: NCHW
    n = 32
    c = 16
    h = 36
    w = 64

    sample_input = torch.randn(n, c, h, w)
    sample_grid = torch.randn(n, h, w, 2)
    sample_output = grid_sampler.forward(sample_input, sample_grid)
    print(f"GridSampler output shape: {sample_output.shape}")

    model_path = os.path.join(os.path.dirname(__file__), "grid_sampler_models")
    os.makedirs(model_path, exist_ok=True)

    onnx_model_file = os.path.join(model_path, "grid_sample_reproduction.onnx")
    tf_model_path = os.path.join(model_path, "grid_sample_reproduction.tf")

    torch.onnx.export(
        GridSamplerOperator(),
        {"input": sample_input, "grid": sample_grid},
        onnx_model_file,
        opset_version=16,
        input_names=["input", "grid"],
        output_names=["output"],
    )

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_file,
        output_folder_path=tf_model_path,
        keep_shape_absolutely_input_names=["input", "grid"],
    )


if __name__ == "__main__":
    main()
