#!/usr/bin/env python3
"""
Export DINOv2 ViT-B/14 to TensorRT engine.

Downloads the HuggingFace DINOv2 model (fully open-source, no auth required),
traces it with torch.jit, exports to ONNX, and builds a TensorRT engine.

Usage:
    python tools/export_dino_v3.py \
        --output resources/models/dino_v2_vit_base.engine \
        --image-size 1024 --patch-size 14 \
        --fp16

Requirements:
    pip install torch transformers onnx onnxruntime tensorrt
    (tensorrt Python package must match installed TensorRT SDK version)
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export DINOv2 to TensorRT engine")
    p.add_argument("--output", required=True, help="Output .engine file path")
    p.add_argument("--image-size", type=int, default=518, help="Input image size (square)")
    p.add_argument("--patch-size", type=int, default=14, help="ViT patch size")
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    p.add_argument("--int8", action="store_true", help="Use INT8 calibration (needs calibrator)")
    p.add_argument("--workspace-size", type=int, default=4096, help="TRT workspace in MiB")
    p.add_argument("--model-name", default="facebook/dinov2-base",
                   help="HuggingFace model ID")
    return p.parse_args()


def build_trt_engine(onnx_path: Path, output_path: Path, args) -> None:
    """Build TensorRT engine from ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}", file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_size * 1024 * 1024)

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16: enabled")
        else:
            print("  FP16: not supported by this GPU, falling back to FP32")

    profile = builder.create_optimization_profile()
    c, h, w = 3, args.image_size, args.image_size
    profile.set_shape("pixel_values", (1, c, h, w), (1, c, h, w), (1, c, h, w))
    config.add_optimization_profile(profile)

    print(f"  Building TensorRT engine (this may take 5-20 minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: TensorRT engine build failed", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {output_path} ({len(serialized) / 1024 / 1024:.1f} MiB)")


def export_onnx(args) -> Path:
    """Load HuggingFace DINOv2 model and export to ONNX."""
    import torch
    from transformers import AutoModel

    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()

    if args.fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    h, w = args.image_size, args.image_size

    # DINOv2 returns BaseModelOutputWithPooling; wrap to extract only
    # last_hidden_state so ONNX export produces a single output tensor.
    class FeatureExtractor(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values):
            return self.backbone(pixel_values).last_hidden_state

    wrapped = FeatureExtractor(model)
    wrapped.eval()

    dummy = torch.randn(1, 3, h, w, dtype=dtype)

    onnx_path = Path(args.output).with_suffix(".onnx")
    print(f"  Exporting to ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            opset_version=17,
            dynamo=False,
        )
    print(f"  ONNX export complete: {onnx_path}")
    return onnx_path


def main():
    args = parse_args()
    print("=" * 60)
    print("DINOv2 → TensorRT Engine Export")
    print(f"  Model:    {args.model_name}")
    print(f"  Image:    {args.image_size}×{args.image_size}")
    print(f"  Precision:{'FP16' if args.fp16 else 'FP32'}")
    print(f"  Output:   {args.output}")
    print("=" * 60)

    onnx_path = export_onnx(args)
    build_trt_engine(onnx_path, Path(args.output), args)
    print("\nDone! Engine ready for use with TensorRtEngine.")


if __name__ == "__main__":
    main()
