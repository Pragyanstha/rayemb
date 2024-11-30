# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/05_metrics.ipynb.

# %% ../notebooks/api/05_metrics.ipynb 3
from __future__ import annotations

import torch
import torch.nn as nn

# %% auto 0
__all__ = ['NormalizedCrossCorrelation2d', 'MultiscaleNormalizedCrossCorrelation2d', 'GradientNormalizedCrossCorrelation2d']

# %% ../notebooks/api/05_metrics.ipynb 4
class NormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images."""

    def __init__(self, patch_size=None, eps=1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(self, x1, x2):
        if self.patch_size is not None:
            x1 = to_patches(x1, self.patch_size)
            x2 = to_patches(x2, self.patch_size)
        assert x1.shape == x2.shape, "Input images must be the same size"
        _, c, h, w = x1.shape
        x1, x2 = self.norm(x1), self.norm(x2)
        score = torch.einsum("b...,b...->b", x1, x2)
        score /= c * h * w
        return score

    def norm(self, x):
        mu = x.mean(dim=[-1, -2], keepdim=True)
        var = x.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
        std = var.sqrt()
        return (x - mu) / std


class MultiscaleNormalizedCrossCorrelation2d(torch.nn.Module):
    """Compute Normalized Cross Correlation between two batches of images at multiple scales."""

    def __init__(self, patch_sizes=[None], patch_weights=[1.0], eps=1e-5):
        super().__init__()

        assert len(patch_sizes) == len(patch_weights), "Each scale must have a weight"
        self.nccs = [
            NormalizedCrossCorrelation2d(patch_size) for patch_size in patch_sizes
        ]
        self.patch_weights = patch_weights

    def forward(self, x1, x2):
        scores = []
        for weight, ncc in zip(self.patch_weights, self.nccs):
            scores.append(weight * ncc(x1, x2))
        return torch.stack(scores, dim=0).sum(dim=0)

# %% ../notebooks/api/05_metrics.ipynb 5
from einops import rearrange


def to_patches(x, patch_size):
    x = x.unfold(2, patch_size, step=1).unfold(3, patch_size, step=1).contiguous()
    return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")

# %% ../notebooks/api/05_metrics.ipynb 6
class GradientNormalizedCrossCorrelation2d(NormalizedCrossCorrelation2d):
    """Compute Normalized Cross Correlation between the image gradients of two batches of images."""

    def __init__(self, patch_size=None, sigma=1.0, **kwargs):
        super().__init__(patch_size, **kwargs)
        self.sobel = Sobel(sigma)

    def forward(self, x1, x2):
        return super().forward(self.sobel(x1), self.sobel(x2))

# %% ../notebooks/api/05_metrics.ipynb 7
from torchvision.transforms.functional import gaussian_blur


class Sobel(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,  # X- and Y-gradients
            kernel_size=3,
            stride=1,
            padding=1,  # Return images of the same size as inputs
            bias=False,
        )

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(torch.float32)
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(torch.float32)
        G = torch.stack([Gx, Gy]).unsqueeze(1)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = gaussian_blur(img, 3, self.sigma)
        x = self.filter(img)
        return x
