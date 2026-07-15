#!/usr/bin/env python3
"""
Export CLIP ViT-B/32 image encoder to TensorRT engine.

Downloads the HuggingFace CLIP vision model, traces it with torch.jit,
exports to ONNX, and builds a TensorRT engine (.engine file).

Usage:
    python tools/export_clip.py \
        --output resources/models/clip_vit_b32.engine \
        --image-size 224 \
        --fp16

Requirements:
    pip install torch transformers onnx onnxruntime tensorrt
    (tensorrt Python package must match installed TensorRT SDK version)
"""

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Export CLIP vision encoder to TensorRT engine")
    p.add_argument("--output", required=True, help="Output .engine file path")
    p.add_argument("--image-size", type=int, default=224, help="Input image size (square)")
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    p.add_argument("--int8", action="store_true", help="Use INT8 calibration (needs calibrator)")
    p.add_argument("--workspace-size", type=int, default=4096, help="TRT workspace in MiB")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32",
                   help="HuggingFace model ID")
    return p.parse_args()


class CLIPVisionWrapper(torch.nn.Module):
    """Wraps CLIPVisionModel so the forward pass returns only the pooled image feature.

    CLIPVisionModel.forward() normally returns BaseModelOutputWithPooling with
    two keys: ``last_hidden_state`` (patch + CLS tokens) and ``pooler_output``
    (projected CLS = 512-dim global feature).  The adapter expects a single
    output tensor named ``features``, so this wrapper strips the dict.
    """

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values).pooler_output


def build_trt_engine(onnx_path: Path, output_path: Path, args) -> None:
    """Build TensorRT engine from ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

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

    print(f"  Building TensorRT engine (this may take 2-10 minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: TensorRT engine build failed", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {output_path} ({len(serialized) / 1024 / 1024:.1f} MiB)")


def export_onnx(args) -> Path:
    """Load HuggingFace CLIP vision model, trace, and export to ONNX."""
    import torch
    from transformers import CLIPVisionModel

    print(f"Loading model: {args.model_name}")
    raw_model = CLIPVisionModel.from_pretrained(args.model_name)
    raw_model.eval()

    if args.fp16:
        raw_model = raw_model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Wrap so forward() returns only the pooled 512-d feature.
    model = CLIPVisionWrapper(raw_model)
    model.eval()

    h, w = args.image_size, args.image_size

    dummy = torch.randn(1, 3, h, w, dtype=dtype)

    print("  Tracing model with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    onnx_path = Path(args.output).with_suffix(".onnx")
    print(f"  Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        traced,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["features"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  ONNX export complete: {onnx_path}")
    return onnx_path


def main():
    args = parse_args()
    print("=" * 60)
    print("CLIP ViT-B/32 → TensorRT Engine Export")
    print(f"  Model:      {args.model_name}")
    print(f"  Image:      {args.image_size}×{args.image_size}")
    print(f"  Precision:  {'FP16' if args.fp16 else 'FP32'}")
    print(f"  Output:     {args.output}")
    print("=" * 60)

    onnx_path = export_onnx(args)
    build_trt_engine(onnx_path, Path(args.output), args)
    print("\nDone! Engine ready for use with ClipAdapter.")


if __name__ == "__main__":
    main()
