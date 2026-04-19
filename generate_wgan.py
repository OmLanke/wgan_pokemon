"""
Generate Pokemon sprites from a trained WGAN-GP generator checkpoint.

Usage
-----
Basic (64 images, 4× upscale, truncation 0.8):
  !python generate_wgan.py \\
      --checkpoint /content/drive/MyDrive/pokemon_checkpoints/checkpoint_final.pt

Custom grid and output path:
  !python generate_wgan.py \\
      --checkpoint /content/drive/MyDrive/pokemon_checkpoints/checkpoint_epoch0300.pt \\
      --num_images 100 \\
      --truncation 0.7 \\
      --upscale 6 \\
      --output /content/drive/MyDrive/pokemon_checkpoints/gen_100_trunc07.png

Truncation guide
----------------
  0.5  → very average/clean shapes, low diversity
  0.7  → good quality/diversity balance  (recommended)
  0.8  → more diverse, slight quality drop
  1.0  → pure random normal, maximum diversity, some artefacts
"""

import argparse
import math
import os

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from model_wgan import Generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate images from a trained WGAN-GP generator"
    )

    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint saved by train_wgan.py",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <checkpoint_dir>/generated_<timestamp>.png",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=64,
        help="Number of images to generate (rounded down to nearest perfect square) [64]",
    )
    p.add_argument(
        "--truncation",
        type=float,
        default=0.8,
        help="z clipping threshold — lower = cleaner, less diverse [0.8]",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="LANCZOS upscale multiplier applied to the output grid [4]",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Pixel padding between grid tiles (before upscale) [2]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix random seed for reproducible outputs",
    )

    # Override architecture only if not stored in checkpoint
    p.add_argument(
        "--z_dim",
        type=int,
        default=None,
        help="Noise dim — read from checkpoint automatically, only override if needed",
    )
    p.add_argument(
        "--ngf",
        type=int,
        default=None,
        help="Generator feature maps — read from checkpoint automatically",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_z(
    n: int, z_dim: int, truncation: float, device: torch.device
) -> torch.Tensor:
    """
    Sample noise vectors with optional truncation.
    Clamps each dimension independently to [-truncation, truncation].
    """
    z = torch.randn(n, z_dim, device=device)
    if truncation < 1.0:
        z = z.clamp(-truncation, truncation)
    return z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device]     {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    ckpt_epoch = ckpt.get("epoch", "?")
    if isinstance(ckpt_epoch, int):
        ckpt_epoch += 1  # stored as 0-indexed

    z_dim = args.z_dim or saved.get("z_dim", 128)
    ngf = args.ngf or saved.get("ngf", 64)

    print(f"[Checkpoint] epoch={ckpt_epoch}  z_dim={z_dim}  ngf={ngf}")

    # ── Build generator ───────────────────────────────────────────────────
    G = Generator(z_dim=z_dim, ngf=ngf).to(device)
    # torch.compile prefixes all keys with "_orig_mod." — strip it so the
    # state dict loads correctly whether the checkpoint was saved from a
    # compiled or uncompiled model.
    state_dict = ckpt["generator"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    G.load_state_dict(state_dict)
    G.eval()

    param_count = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"[Generator]  {param_count:.2f}M parameters")

    # ── Sample grid dimensions ────────────────────────────────────────────
    grid_side = int(math.floor(math.sqrt(args.num_images)))
    num_images = grid_side * grid_side
    if num_images != args.num_images:
        print(
            f"[Grid]  Rounded {args.num_images} → {num_images} ({grid_side}×{grid_side})"
        )
    else:
        print(f"[Grid]  {num_images} images ({grid_side}×{grid_side})")

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"[Sampling]   truncation={args.truncation}")
    z = sample_z(num_images, z_dim, args.truncation, device)

    with torch.no_grad():
        fake = G(z).cpu().float()

    # ── Build grid image ──────────────────────────────────────────────────
    grid = vutils.make_grid(
        fake, nrow=grid_side, normalize=True, value_range=(-1, 1), padding=args.padding
    )
    img_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    if args.upscale > 1:
        pil_img = pil_img.resize(
            (pil_img.width * args.upscale, pil_img.height * args.upscale),
            Image.LANCZOS,
        )

    # ── Resolve output path ───────────────────────────────────────────────
    if args.output is None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        out_path = os.path.join(out_dir, f"generated_{ts}.png")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    pil_img.save(out_path)

    print(f"\n[Saved]  {out_path}")
    print(
        f"         {pil_img.width}×{pil_img.height} px  "
        f"({num_images} sprites, {args.upscale}× upscale)"
    )


if __name__ == "__main__":
    main()
