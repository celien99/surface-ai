#!/usr/bin/env python3
"""
Export SAM2 (Segment Anything Model 2) to TensorRT engine.

Downloads SAM2 from Meta Research, wraps the image encoder + prompt encoder +
mask decoder into a single traceable module, exports to ONNX, and builds a
TensorRT engine.

Usage:
    python tools/export_sam2.py \
        --output resources/models/sam2_vit_h.engine \
        --image-size 1024 \
        --fp16

Requirements:
    pip install torch onnx onnxruntime tensorrt
    pip install git+https://github.com/facebookresearch/sam2.git

    The SAM2 checkpoint will be downloaded automatically on first run.
"""

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Export SAM2 to TensorRT engine")
    p.add_argument("--output", required=True, help="Output .engine file path")
    p.add_argument("--image-size", type=int, default=1024, help="Input image size (square)")
    p.add_argument("--mask-size", type=int, default=256, help="Output mask size (square)")
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    p.add_argument("--int8", action="store_true", help="Use INT8 calibration (needs calibrator)")
    p.add_argument("--workspace-size", type=int, default=8192, help="TRT workspace in MiB")
    p.add_argument("--model-cfg", default="sam2_hiera_l.yaml",
                   help="SAM2 model config name (resolved from sam2 package)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to SAM2 checkpoint (.pt). Downloads automatically if omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Wrapper: chains image_encoder → prompt_encoder (mask-only) → mask_decoder
# so the full pipeline can be exported as a single ONNX graph.
# ---------------------------------------------------------------------------
class SAM2FullWrapper(torch.nn.Module):
    """Traceable wrapper around the SAM2 inference pipeline.

    Combines the three SAM2 sub-modules so that a single ``forward()`` call
    goes from **(image, mask_prompt) → low_res_masks**, matching the binding
    names expected by ``Sam2Adapter`` (*image* + *prompt* → *masks*).

    .. note::
        Only **mask prompts** are supported at this stage (consistent with the
        Sam2Adapter placeholder in M3).  Point / box prompts will be added in
        M5.
    """

    def __init__(self, sam2_model):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.sam_prompt_encoder = sam2_model.sam_prompt_encoder
        self.mask_decoder = sam2_model.mask_decoder

        # SAM2Base computes backbone feature sizes during set_image().
        # We capture the internal helper so the wrapper can do the same
        # post-processing the mask_decoder expects.
        self._bb_feat_sizes = None  # set on first forward

    def _prepare_image_features(self, backbone_out):
        """Prepare backbone features for the mask decoder.

        SAM2's internal pipeline runs image_encoder → (optional neck/FPN) →
        flatten + permute → image_embeddings + image_pe.  This wrapper
        replicates the logic from ``SAM2Base._prepare_backbone_features()``
        and ``SAM2Base._prepare_backbone_features_hiera()``.

        If the installed SAM2 version uses a different internal method name,
        adjust the forwarding below.
        """
        # Hiera backbone returns a dict with 'vision_features' (list of
        # tensors at different scales) and 'vision_pos_enc' (list of
        # positional encodings).
        if isinstance(backbone_out, dict):
            vision_features = backbone_out["vision_features"]
            vision_pos_enc = backbone_out["vision_pos_enc"]
        else:
            # Some backbones return a plain list / tensor.
            vision_features = backbone_out
            vision_pos_enc = [None] * len(vision_features)

        # Flatten spatial dims and concatenate across feature levels.
        # For Hiera: each feat has shape [B, C, H, W] → [B, H*W, C].
        flat_features = []
        flat_pos_enc = []
        for feat, pe in zip(vision_features, vision_pos_enc):
            b, c, h, w = feat.shape
            flat_features.append(feat.permute(0, 2, 3, 1).reshape(b, h * w, c))
            if pe is not None:
                b_pe, c_pe, h_pe, w_pe = pe.shape
                flat_pos_enc.append(pe.permute(0, 2, 3, 1).reshape(b_pe, h_pe * w_pe, c_pe))

        image_embeddings = torch.cat(flat_features, dim=1)
        if flat_pos_enc:
            image_pe = torch.cat(flat_pos_enc, dim=1)
        else:
            # Fallback: zero positional encoding (some backbones bake PE into
            # features directly).
            image_pe = torch.zeros_like(image_embeddings)

        return image_embeddings, image_pe

    def forward(self, image, prompt):
        """Full SAM2 forward pass (image + mask prompt → masks).

        Args:
            image:  [B, 3, H, W] float32 / float16 input image.
            prompt: [B, 4, H_p, W_p] float32 / float16 mask prompt
                    (4 channels encode the down-sampled mask at different
                    scales, matching SAM2's mask prompt convention).

        Returns:
            masks: [B, 1, mask_size, mask_size] predicted masks.
        """
        # 1. Image encode
        backbone_out = self.image_encoder(image)

        # 2. Prepare features for mask decoder
        image_embeddings, image_pe = self._prepare_image_features(backbone_out)

        # 3. Prompt encode (mask-only prompt → no sparse embeddings)
        sparse_emb, dense_emb = self.sam_prompt_encoder(
            points=(None, None),
            boxes=None,
            masks=prompt,
        )

        # 4. Mask decode
        low_res_masks, _iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        return low_res_masks


# ---------------------------------------------------------------------------
# TensorRT builder
# ---------------------------------------------------------------------------
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

    # Optimization profile: fixed batch = 1 for inference.
    profile = builder.create_optimization_profile()
    c, h, w = 3, args.image_size, args.image_size
    mp = args.mask_size
    profile.set_shape("image",  (1, c, h, w),  (1, c, h, w),  (1, c, h, w))
    profile.set_shape("prompt", (1, 4, mp, mp), (1, 4, mp, mp), (1, 4, mp, mp))
    config.add_optimization_profile(profile)

    print("  Building TensorRT engine (this may take 10-30 minutes for SAM2)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: TensorRT engine build failed", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {output_path} ({len(serialized) / 1024 / 1024:.1f} MiB)")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(args) -> Path:
    """Build SAM2 model, wrap, and export to ONNX."""
    from sam2.build_sam import build_sam2

    mask_size = args.mask_size
    image_size = args.image_size

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Auto-download: build_sam2 with no explicit checkpoint will download
        # from Meta's CDN into the default cache directory.
        checkpoint = None  # build_sam2 handles this

    print(f"Building SAM2 model: {args.model_cfg}")
    sam2 = build_sam2(args.model_cfg, checkpoint)
    sam2.eval()

    if args.fp16:
        sam2 = sam2.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    wrapper = SAM2FullWrapper(sam2)
    wrapper.eval()

    h, w = image_size, image_size
    dummy_image = torch.randn(1, 3, h, w, dtype=dtype)
    dummy_prompt = torch.randn(1, 4, mask_size, mask_size, dtype=dtype)

    print("  Exporting SAM2 full pipeline to ONNX via torch.onnx.export...")
    print("  (This uses symbolic tracing — if it fails on control-flow ops,")
    print("   try simplifying the wrapper._prepare_image_features method.)")

    onnx_path = Path(args.output).with_suffix(".onnx")

    # Use torch.onnx.export directly (not torch.jit.trace) so that dynamic
    # control flow inside the mask decoder is captured symbolically.
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_prompt),
        str(onnx_path),
        input_names=["image", "prompt"],
        output_names=["masks"],
        dynamic_axes={
            "image":  {0: "batch"},
            "prompt": {0: "batch"},
            "masks":  {0: "batch"},
        },
        opset_version=17,
        # SAM2's mask decoder has dropout / training guards — force eval mode.
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"  ONNX export complete: {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    print("=" * 60)
    print("SAM2 → TensorRT Engine Export")
    print(f"  Config:     {args.model_cfg}")
    print(f"  Image:      {args.image_size}×{args.image_size}")
    print(f"  Mask:       {args.mask_size}×{args.mask_size}")
    print(f"  Precision:  {'FP16' if args.fp16 else 'FP32'}")
    print(f"  Output:     {args.output}")
    print("=" * 60)

    onnx_path = export_onnx(args)
    build_trt_engine(onnx_path, Path(args.output), args)
    print("\nDone! Engine ready for use with Sam2Adapter.")


if __name__ == "__main__":
    main()
