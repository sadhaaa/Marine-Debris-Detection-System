"""
Underwater-aware Batch Normalization (UWBN)
From: AquaYOLO26: A Degradation-Aware YOLO26 Framework for Underwater Marine Debris Detection
      (Zeng et al., Symmetry 2026)

Key mechanics:
1. Wavelength-dependent light attenuation follows Beer-Lambert law:
   I_c(d) = I_0 * exp(-alpha_c * d)
   alpha_red ≈ 0.80 m^-1, alpha_green ≈ 0.04 m^-1, alpha_blue ≈ 0.01 m^-1
2. Deep features partition channels into 3 groups (G1, G2, G3) corresponding to color cues.
3. Soft depth proxy d_hat is estimated from group mean magnitudes:
   d_hat = softplus(- (1/3) * sum(x_Gk / x_ref))
4. Pre-compensation before BN:
   x_tilde = x * exp(alpha_hat_c * d_hat)
   where alpha_hat = W_alpha * alpha_phys + b_alpha
5. Standard BatchNorm on x_tilde.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UWBN(nn.Module):
    """
    Underwater-aware Batch Normalization.
    Pre-compensates for wavelength-dependent color attenuation before standard BN.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Physical attenuation priors: [red, green, blue]
        self.register_buffer('alpha_phys', torch.tensor([0.80, 0.04, 0.01], dtype=torch.float32))

        # Linear projection parameters to get alpha_hat per channel: W_alpha in R^(C x 3), b_alpha in R^C
        self.W_alpha = nn.Parameter(torch.randn(num_features, 3) * 0.02)
        self.b_alpha = nn.Parameter(torch.zeros(num_features))

        # Running reference magnitude across training batches
        self.register_buffer('x_ref', torch.tensor(1.0, dtype=torch.float32))

        # Standard BN affine parameters & running stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. Group channels into 3 contiguous groups (spectral proxies)
        Cg = max(1, C // 3)
        g1 = x[:, :Cg, :, :]
        g2 = x[:, Cg:2*Cg, :, :]
        g3 = x[:, 2*Cg:, :, :]

        mean_g1 = torch.mean(g1, dim=(1, 2, 3))  # [B]
        mean_g2 = torch.mean(g2, dim=(1, 2, 3))  # [B]
        mean_g3 = torch.mean(g3, dim=(1, 2, 3))  # [B]
        group_mean = (mean_g1 + mean_g2 + mean_g3) / 3.0  # [B]

        if self.training:
            # Update running x_ref
            batch_mag = torch.mean(torch.abs(x)).detach()
            self.x_ref.copy_((1.0 - self.momentum) * self.x_ref + self.momentum * batch_mag)

        ref = torch.clamp(self.x_ref, min=1e-4)
        # Soft depth proxy d_hat > 0
        d_hat = F.softplus(-group_mean / ref).view(B, 1, 1, 1)  # [B, 1, 1, 1]

        # 2. Learnable attenuation per channel
        # alpha_hat = W_alpha * alpha_phys + b_alpha  -> [C]
        alpha_hat = torch.matmul(self.W_alpha, self.alpha_phys) + self.b_alpha
        # Bound attenuation compensation factor to prevent explosion
        alpha_hat = torch.clamp(alpha_hat, min=-2.0, max=2.0).view(1, C, 1, 1)

        # 3. Pre-compensation: x_tilde = x * exp(alpha_hat * d_hat)
        # Clamp exponent for numeric stability
        compensation = torch.exp(torch.clamp(alpha_hat * d_hat, max=2.0))
        x_tilde = x * compensation

        # 4. Standard Batch Normalization on compensated features
        return F.batch_norm(
            x_tilde,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.num_batches_tracked.bool(),
            self.momentum,
            self.eps
        )
