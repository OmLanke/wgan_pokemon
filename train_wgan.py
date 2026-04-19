"""
WGAN-GP training script for Pokemon sprite generation.

Quickstart (Google Colab T4)
-----------------------------
# Mount Drive first, then:
!python train_wgan.py \\
    --data_dir      /content/drive/MyDrive/pokemon_dataset \\
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \\
    --epochs 500

Resume from a checkpoint:
!python train_wgan.py \\
    --data_dir      /content/drive/MyDrive/pokemon_dataset \\
    --checkpoint_dir /content/drive/MyDrive/pokemon_checkpoints \\
    --resume /content/drive/MyDrive/pokemon_checkpoints/checkpoint_epoch0300.pt \\
    --epochs 500

Speed tips
----------
- Default batch_size=128 fits comfortably on a T4 (16 GB VRAM) with AMP.
- Add --compile for an extra ~15% speedup via torch.compile (PyTorch 2.0+).
- Reduce --n_critic to 3 if you want faster epochs at slightly lower quality.
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model_wgan import Critic, Generator, gradient_penalty, weights_init


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train WGAN-GP on Pokemon sprites")

    # Paths
    p.add_argument(
        "--data_dir",
        default="/content/drive/MyDrive/pokemon_dataset",
        help="Folder containing PNG/JPG training images",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="/content/drive/MyDrive/pokemon_checkpoints",
        help="Folder to save checkpoints and sample grids",
    )

    # Architecture
    p.add_argument(
        "--image_size", type=int, default=64, help="Training image resolution [64]"
    )
    p.add_argument(
        "--z_dim", type=int, default=128, help="Noise vector dimensionality [128]"
    )
    p.add_argument(
        "--ngf", type=int, default=64, help="Generator base feature-map count [64]"
    )
    p.add_argument(
        "--ndf", type=int, default=64, help="Critic base feature-map count [64]"
    )

    # Training
    p.add_argument(
        "--epochs", type=int, default=500, help="Total training epochs [500]"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size — 128 fits T4 comfortably with AMP [128]",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate for both G and C [1e-4]",
    )
    p.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="Critic updates per generator update [5]",
    )
    p.add_argument(
        "--lambda_gp", type=float, default=10.0, help="Gradient penalty weight [10.0]"
    )
    p.add_argument(
        "--lazy_reg",
        type=int,
        default=4,
        help="Compute gradient penalty every N critic steps (lazy regularisation). "
        "lambda_gp is scaled up by N automatically. Set to 1 to disable [4]",
    )

    # Logging / saving
    p.add_argument(
        "--sample_interval",
        type=int,
        default=25,
        help="Save a sample image grid every N epochs [25]",
    )
    p.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save a checkpoint every N epochs [50]",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=4,
        help="LANCZOS upscale factor for saved sample grids [4]",
    )

    # Misc
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to resume training from",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Enable torch.compile — on by default, pass --no_compile to disable",
    )
    p.add_argument(
        "--no_compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed [42]")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PokemonDataset(Dataset):
    """
    Loads PNG/JPG Pokemon sprites.

    PNGs with transparency are composited onto a white background before
    resizing so the generator learns clean sprites rather than noise on
    transparent pixels.
    """

    EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

    def __init__(self, data_dir: str, image_size: int = 64):
        self.paths = []
        for ext in self.EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(data_dir, ext)))
        self.paths.sort()

        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {data_dir!r}. "
                "Check the path and that the folder contains PNG or JPG files."
            )
        print(f"[Dataset] {len(self.paths)} images  ({data_dir})")

        self.transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.LANCZOS
                ),
                # Augmentation — applied every epoch so each pass sees different
                # versions of the 819 sprites. Keeps training time identical
                # (CPU-side, overlapped with GPU compute) but greatly improves
                # generalisation and output variety.
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),  # → [0, 1]
                T.Normalize(
                    [0.5, 0.5, 0.5],  # → [-1, 1]
                    [0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx])
        # Composite RGBA → white RGB so transparent areas become white background
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_sample_grid(
    generator: torch.nn.Module,
    z_fixed: torch.Tensor,
    epoch: int,
    out_dir: str,
    upscale: int = 4,
) -> str:
    """Generate a fixed-seed grid image and save it to out_dir."""
    generator.eval()
    with torch.no_grad():
        fake = generator(z_fixed).cpu().float()
    grid = vutils.make_grid(
        fake, nrow=8, normalize=True, value_range=(-1, 1), padding=2
    )
    img_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    if upscale > 1:
        pil_img = pil_img.resize(
            (pil_img.width * upscale, pil_img.height * upscale),
            Image.LANCZOS,
        )
    out_path = os.path.join(out_dir, f"sample_epoch_{epoch:04d}.png")
    pil_img.save(out_path)
    generator.train()
    return out_path


def save_checkpoint(
    path: str,
    epoch: int,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "opt_G": opt_g.state_dict(),
            "opt_C": opt_c.state_dict(),
            "args": vars(args),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample_dir = os.path.join(args.checkpoint_dir, "samples")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        cudnn.benchmark = True  # auto-tune conv kernels for fixed input size
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU]    {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("[Device] CPU — training will be slow. Connect a GPU runtime.")
    print(f"[AMP]    {'enabled (float16 forward passes)' if use_amp else 'disabled'}")

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = PokemonDataset(args.data_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        drop_last=True,  # keep batch sizes consistent for gradient penalty
    )

    # ── Models ────────────────────────────────────────────────────────────
    G = Generator(z_dim=args.z_dim, ngf=args.ngf).to(device)
    C = Critic(ndf=args.ndf).to(device)
    G.apply(weights_init)
    C.apply(weights_init)

    g_params = sum(p.numel() for p in G.parameters())
    c_params = sum(p.numel() for p in C.parameters())
    print(
        f"[Model]  Generator {g_params / 1e6:.2f}M params | "
        f"Critic {c_params / 1e6:.2f}M params"
    )

    # ── Optimisers ────────────────────────────────────────────────────────
    # betas=(0, 0.9) recommended by WGAN-GP paper (no first-moment momentum)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["generator"])
        C.load_state_dict(ckpt["critic"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_C.load_state_dict(ckpt["opt_C"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Loaded checkpoint — resuming from epoch {start_epoch}")

    # ── torch.compile ─────────────────────────────────────────────────────
    # Only compile the Generator. The Critic is called inside gradient_penalty
    # with create_graph=True (higher-order autograd). torch.compile uses AOT
    # Autograd internally, which conflicts with create_graph=True and causes
    # an inplace-op error on backward — regardless of whether C_for_gp tricks
    # are used, because torch.compile wraps the module in-place so both
    # references point to the same compiled graph.
    C_for_gp = C  # same object as C — kept for clarity in the training loop
    if args.compile:
        try:
            G = torch.compile(G)
            print(
                "[Compile] torch.compile enabled (Generator only — Critic must stay eager for WGAN-GP)"
            )
        except Exception as exc:
            print(f"[Compile] torch.compile unavailable: {exc}")

    # ── Fixed noise for consistent sample grids ───────────────────────────
    z_fixed = torch.randn(64, args.z_dim, device=device)

    # ── Training summary ──────────────────────────────────────────────────
    batches_per_epoch = len(loader)
    total_iters = (args.epochs - start_epoch) * batches_per_epoch
    steps_per_iter = args.n_critic + 1
    print(
        f"\n[Train]  epochs={args.epochs}  batch={args.batch_size}  "
        f"batches/epoch={batches_per_epoch}\n"
        f"         n_critic={args.n_critic}  lambda_gp={args.lambda_gp}  "
        f"lazy_reg={args.lazy_reg}  z_dim={args.z_dim}  lr={args.lr}\n"
        f"         ~{total_iters * steps_per_iter} optimizer steps total\n"
    )

    # Save untrained sample so you can see the baseline
    out = save_sample_grid(G, z_fixed, 0, sample_dir, args.upscale)
    print(f"[Init]   Saved baseline sample → {out}\n")

    t_train_start = time.time()
    global_critic_step = 0  # for lazy regularisation

    for epoch in range(start_epoch, args.epochs):
        G.train()
        C.train()
        epoch_c_loss = 0.0
        epoch_g_loss = 0.0
        t_epoch = time.time()

        for real in loader:
            real = real.to(device, non_blocking=True)
            batch_size = real.size(0)

            # ── Critic steps ──────────────────────────────────────────────
            # No autocast: gradient penalty requires float32 for numerically
            # stable higher-order autograd. Mixing autocast with create_graph=True
            # causes inplace-op conflicts even in eager mode.
            #
            # Lazy regularisation (StyleGAN2): compute the gradient penalty only
            # every `lazy_reg` critic steps and scale lambda accordingly.
            # Default lazy_reg=4 → GP computed 25% of the time → ~3× critic speedup
            # with no meaningful quality loss.
            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.z_dim, device=device)

                fake = G(z).detach()
                c_real = C(real).mean()
                c_fake = C(fake).mean()
                c_loss_w = c_fake - c_real  # Wasserstein estimate (maximise real−fake)

                c_loss = c_loss_w
                if global_critic_step % args.lazy_reg == 0:
                    # C_for_gp is the uncompiled Critic — required for
                    # create_graph=True to work with torch.compile on C.
                    gp = gradient_penalty(C_for_gp, real, fake, device)
                    # Scale lambda by lazy_reg to keep effective regularisation
                    # strength constant regardless of how often GP is applied.
                    c_loss = c_loss_w + (args.lambda_gp * args.lazy_reg) * gp

                opt_C.zero_grad(set_to_none=True)
                c_loss.backward()
                opt_C.step()
                global_critic_step += 1

            epoch_c_loss += c_loss.item()

            # ── Generator step ────────────────────────────────────────────
            z = torch.randn(batch_size, args.z_dim, device=device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                fake = G(z)
                g_loss = -C(fake).mean()

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            epoch_g_loss += g_loss.item()

        # ── End-of-epoch logging ──────────────────────────────────────────
        nb = batches_per_epoch
        epoch_time = time.time() - t_epoch
        elapsed = time.time() - t_train_start
        eta = (args.epochs - epoch - 1) * epoch_time

        print(
            f"Epoch [{epoch + 1:4d}/{args.epochs}]  "
            f"C: {epoch_c_loss / nb:+7.4f}  G: {epoch_g_loss / nb:+7.4f}  "
            f"| {epoch_time:.1f}s/epoch  "
            f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m"
        )

        # ── Sample grid ───────────────────────────────────────────────────
        if (epoch + 1) % args.sample_interval == 0:
            out = save_sample_grid(G, z_fixed, epoch + 1, sample_dir, args.upscale)
            print(f"  → sample  {out}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch{epoch + 1:04d}.pt"
            )
            save_checkpoint(ckpt_path, epoch, G, C, opt_G, opt_C, args)
            print(f"  → checkpoint  {ckpt_path}")

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, args.epochs - 1, G, C, opt_G, opt_C, args)

    total_time = time.time() - t_train_start
    print(f"\n[Done]  Training complete in {total_time / 60:.1f} min")
    print(f"[Done]  Final checkpoint → {final_path}")
    print(f"[Done]  Sample grids     → {sample_dir}/")


if __name__ == "__main__":
    train()
