"""
WGAN-GP model definitions: Generator and Critic.

Architecture summary
--------------------
Generator  : z (z_dim,) → Linear → reshape 4×4×(ngf*8)
             → 4× ConvTranspose2d blocks → 64×64×3  (Tanh)
Critic     : 64×64×3 → 4× Conv2d blocks (SpectralNorm, no BN)
             → Flatten → Linear → scalar  (no sigmoid)

Notes
-----
- No BatchNorm in Critic — required for WGAN-GP (gradient penalty
  assumes per-sample independence across the batch).
- SpectralNorm on all Critic layers adds a soft Lipschitz constraint
  on top of the gradient penalty for extra stability.
- weights_init applies the DCGAN paper initialisation (N(0, 0.02)).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def weights_init(m):
    classname = type(m).__name__
    if classname in ("Conv2d", "ConvTranspose2d"):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname == "Linear":
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname == "BatchNorm2d":
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """
    Maps a latent vector z ∈ R^z_dim to a 3×64×64 image in [-1, 1].

    Layer progression (ngf=64 default):
      Linear           →  (ngf*8) × 4 × 4   = 512 × 4 × 4
      ConvTranspose 1  →  (ngf*4) × 8 × 8   = 256 × 8 × 8
      ConvTranspose 2  →  (ngf*2) × 16 × 16 = 128 × 16 × 16
      ConvTranspose 3  →  (ngf)   × 32 × 32 =  64 × 32 × 32
      ConvTranspose 4  →  3       × 64 × 64
    """

    def __init__(self, z_dim: int = 128, ngf: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.ngf = ngf

        self.project = nn.Sequential(
            nn.Linear(z_dim, ngf * 8 * 4 * 4, bias=False),
        )

        self.conv_blocks = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(
                ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=False),
            # 8×8 → 16×16
            nn.ConvTranspose2d(
                ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
            # 16×16 → 32×32
            nn.ConvTranspose2d(
                ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=False),
            # 32×32 → 64×64
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(x.size(0), self.ngf * 8, 4, 4)
        return self.conv_blocks(x)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class Critic(nn.Module):
    """
    Maps a 3×64×64 image to a scalar Wasserstein score (higher = more real).

    Layer progression (ndf=64 default):
      Conv 1  →  (ndf)   × 32 × 32 =  64 × 32 × 32
      Conv 2  →  (ndf*2) × 16 × 16 = 128 × 16 × 16
      Conv 3  →  (ndf*4) × 8 × 8   = 256 × 8 × 8
      Conv 4  →  (ndf*8) × 4 × 4   = 512 × 4 × 4
      Linear  →  1 (scalar)

    All convolutions wrapped with SpectralNorm.
    No BatchNorm (incompatible with per-sample gradient penalty).
    """

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.ndf = ndf

        self.net = nn.Sequential(
            # 64×64 → 32×32
            spectral_norm(
                nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=False),
            # 32×32 → 16×16
            spectral_norm(
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=False),
            # 16×16 → 8×8
            spectral_norm(
                nn.Conv2d(
                    ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=False),
            # 8×8 → 4×4
            spectral_norm(
                nn.Conv2d(
                    ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=False),
            # 4×4 → scalar
            nn.Flatten(),
            spectral_norm(nn.Linear(ndf * 8 * 4 * 4, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes the WGAN-GP gradient penalty (Gulrajani et al. 2017).

    Samples a random linear interpolation between real and fake images,
    runs it through the critic, and penalises deviation from unit gradient norm.

    Always runs in float32 regardless of AMP context — required for
    numerically stable autograd through the gradient norm computation.

    Args:
        critic : the Critic module
        real   : batch of real images,  shape (N, 3, H, W), float32
        fake   : batch of fake images,  shape (N, 3, H, W), float32
        device : torch device

    Returns:
        Scalar gradient penalty tensor.
    """
    batch_size = real.size(0)
    # Random mixing coefficient per sample
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, dtype=torch.float32)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    with torch.autocast(device_type=device.type, enabled=False):
        critic_out = critic(interpolated.to(next(critic.parameters()).dtype))

    gradients = torch.autograd.grad(
        outputs=critic_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return penalty
