from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def mse_01(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference.detach().float().clamp(0.0, 1.0)
    estimate = estimate.detach().float().clamp(0.0, 1.0)
    return float(F.mse_loss(estimate, reference).item())


def psnr_01(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = mse_01(reference, estimate)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=torch.float32) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_01(reference: torch.Tensor, estimate: torch.Tensor, window_size: int = 7, sigma: float = 1.5) -> float:
    """Windowed SSIM for tensors shaped [N,C,H,W] in [0,1]."""
    reference = reference.detach().float().clamp(0.0, 1.0)
    estimate = estimate.detach().float().clamp(0.0, 1.0)
    if reference.ndim != 4 or estimate.shape != reference.shape:
        raise ValueError("SSIM inputs must have the same [N,C,H,W] shape.")

    channels = reference.size(1)
    kernel = _gaussian_kernel(window_size, sigma, channels, reference.device)
    padding = window_size // 2
    mu_x = F.conv2d(reference, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(estimate, kernel, padding=padding, groups=channels)
    sigma_x = F.conv2d(reference * reference, kernel, padding=padding, groups=channels) - mu_x**2
    sigma_y = F.conv2d(estimate * estimate, kernel, padding=padding, groups=channels) - mu_y**2
    sigma_xy = F.conv2d(reference * estimate, kernel, padding=padding, groups=channels) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator.clamp_min(1e-12)).mean().item())


class OptionalLPIPS:
    def __init__(self, device: torch.device, enabled: bool = True):
        self.model = None
        if not enabled:
            return
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            self.model = None

    @torch.no_grad()
    def __call__(self, reference: torch.Tensor, estimate: torch.Tensor) -> Optional[float]:
        if self.model is None:
            return None
        reference_scaled = reference.detach().float().clamp(0.0, 1.0) * 2.0 - 1.0
        estimate_scaled = estimate.detach().float().clamp(0.0, 1.0) * 2.0 - 1.0
        value = self.model(reference_scaled, estimate_scaled)
        return float(value.mean().item())
