# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/00_drr.ipynb.

# %% ../notebooks/api/00_drr.ipynb 3
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from fastcore.basics import patch

from .detector import Detector
from .renderers import Siddon, Trilinear
from .pose import convert

# %% auto 0
__all__ = ['DRR', 'Registration']

# %% ../notebooks/api/00_drr.ipynb 7
class DRR(nn.Module):
    """PyTorch module that computes differentiable digitally reconstructed radiographs."""

    def __init__(
        self,
        volume: np.ndarray,  # CT volume
        spacing: np.ndarray,  # Dimensions of voxels in the CT volume
        sdr: float,  # Source-to-detector radius for the C-arm (half of the source-to-detector distance)
        height: int,  # Height of the rendered DRR
        delx: float,  # X-axis pixel size
        width: int | None = None,  # Width of the rendered DRR (default to `height`)
        dely: float | None = None,  # Y-axis pixel size (if not provided, set to `delx`)
        x0: float = 0.0,  # Principal point X-offset
        y0: float = 0.0,  # Principal point Y-offset
        p_subsample: float | None = None,  # Proportion of pixels to randomly subsample
        reshape: bool = True,  # Return DRR with shape (b, 1, h, w)
        reverse_x_axis: bool = False,  # If pose includes reflection (in E(3) not SE(3)), reverse x-axis
        patch_size: int | None = None,  # Render patches of the DRR in series
        bone_attenuation_multiplier: float = 1.0,  # Contrast ratio of bone to soft tissue
        renderer: str = "siddon",  # Rendering backend, either "siddon" or "trilinear"
        **renderer_kwargs,  # Kwargs for the renderer
    ):
        super().__init__()

        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else dely
        n_subsample = (
            int(height * width * p_subsample) if p_subsample is not None else None
        )
        self.detector = Detector(
            sdr,
            height,
            width,
            delx,
            dely,
            x0,
            y0,
            n_subsample=n_subsample,
            reverse_x_axis=reverse_x_axis,
        )

        # Initialize the volume
        self.register_buffer("spacing", torch.tensor(spacing))
        self.register_buffer("volume", torch.tensor(volume).flip([0]))
        self.reshape = reshape
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.n_patches = (height * width) // (self.patch_size**2)

        # Parameters for segmenting the CT volume and reweighting voxels
        self.air = torch.where(self.volume <= -800)
        self.soft_tissue = torch.where((-800 < self.volume) & (self.volume <= 350))
        self.bone = torch.where(350 < self.volume)
        self.bone_attenuation_multiplier = bone_attenuation_multiplier

        # Initialize the renderer
        if renderer == "siddon":
            self.renderer = Siddon(**renderer_kwargs)
        elif renderer == "trilinear":
            self.renderer = Trilinear(**renderer_kwargs)
        else:
            raise ValueError(f"renderer must be 'siddon', not {renderer}")

    def reshape_transform(self, img, batch_size):
        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, 1, self.detector.height, self.detector.width)
            else:
                img = reshape_subsampled_drr(img, self.detector, batch_size)
        return img

    def compute_projection_matrix(self, pose: RigidTransform):
        """Compute the projection matrix for the detector given its pose."""
        intrinsic_matrix = self.detector.intrinsic

        # Compute extrinsic matrix
        extrinsic = (
            pose.inverse().compose(self.detector.translate).compose(self.detector.flip_xz)
        ).matrix

        # Only take the first 3 rows for [R | t]
        extrinsic_matrix = extrinsic[0, :3]

        # Compute projection matrix
        projection_matrix = intrinsic_matrix @ extrinsic_matrix
        return projection_matrix

# %% ../notebooks/api/00_drr.ipynb 8
def reshape_subsampled_drr(
    img: torch.Tensor,
    detector: Detector,
    batch_size: int,
):
    n_points = detector.height * detector.width
    drr = torch.zeros(batch_size, n_points).to(img)
    drr[:, detector.subsamples[-1]] = img
    drr = drr.view(batch_size, 1, detector.height, detector.width)
    return drr

# %% ../notebooks/api/00_drr.ipynb 10
from .pose import convert


@patch
def forward(
    self: DRR,
    *args,  # Some batched representation of SE(3)
    parameterization: str = None,  # Specifies the representation of the rotation
    convention: str = None,  # If parameterization is Euler angles, specify convention
    bone_attenuation_multiplier: float = None,  # Contrast ratio of bone to soft tissue
    **kwargs,  # Passed to the renderer
):
    """Generate DRR with rotational and translational parameters."""
    if not hasattr(self, "density"):
        self.set_bone_attenuation_multiplier(self.bone_attenuation_multiplier)
    if bone_attenuation_multiplier is not None:
        self.set_bone_attenuation_multiplier(bone_attenuation_multiplier)

    if parameterization is None:
        pose = args[0]
    else:
        pose = convert(*args, parameterization=parameterization, convention=convention)
    source, target = self.detector(pose)

    if self.patch_size is not None:
        n_points = target.shape[1] // self.n_patches
        img = []
        for idx in range(self.n_patches):
            t = target[:, idx * n_points : (idx + 1) * n_points]
            partial = self.renderer(self.density, self.spacing, source, t, **kwargs)
            img.append(partial)
        img = torch.cat(img, dim=1)
    else:
        img = self.renderer(self.density, self.spacing, source, target, **kwargs)
    return self.reshape_transform(img, batch_size=len(pose))

# %% ../notebooks/api/00_drr.ipynb 11
@patch
def set_bone_attenuation_multiplier(self: DRR, bone_attenuation_multiplier: float):
    self.density = torch.empty_like(self.volume)
    self.density[self.air] = self.volume[self.soft_tissue].min()
    self.density[self.soft_tissue] = self.volume[self.soft_tissue]
    self.density[self.bone] = self.volume[self.bone] * bone_attenuation_multiplier
    self.density -= self.density.min()
    self.density /= self.density.max()
    self.bone_attenuation_multiplier = bone_attenuation_multiplier

# %% ../notebooks/api/00_drr.ipynb 12
@patch
def set_intrinsics(
    self: DRR,
    sdr: float = None,
    delx: float = None,
    dely: float = None,
    x0: float = None,
    y0: float = None,
):
    self.detector = Detector(
        sdr if sdr is not None else self.detector.sdr,
        self.detector.height,
        self.detector.width,
        delx if delx is not None else self.detector.delx,
        dely if dely is not None else self.detector.dely,
        x0 if x0 is not None else self.detector.x0,
        y0 if y0 is not None else self.detector.y0,
        n_subsample=self.detector.n_subsample,
        reverse_x_axis=self.detector.reverse_x_axis,
    ).to(self.volume)

# %% ../notebooks/api/00_drr.ipynb 13
from .pose import RigidTransform


@patch
def perspective_projection(
    self: DRR,
    pose: RigidTransform,
    pts: torch.Tensor,
):
    extrinsic = (
        pose.inverse().compose(self.detector.translate).compose(self.detector.flip_xz)
    )
    x = extrinsic(pts)
    x = torch.einsum("ij, bnj -> bni", self.detector.intrinsic, x)
    z = x[..., -1].unsqueeze(-1).clone()
    x = x / z
    return x[..., :2]

# %% ../notebooks/api/00_drr.ipynb 14
from torch.nn.functional import pad


@patch
def inverse_projection(
    self: DRR,
    pose: RigidTransform,
    pts: torch.Tensor,
):
    extrinsic = (
        self.detector.flip_xz.inverse()
        .compose(self.detector.translate.inverse())
        .compose(pose)
    )
    x = (
        -2
        * self.detector.sdr
        * torch.einsum(
            "ij, bnj -> bni",
            self.detector.intrinsic.inverse(),
            pad(pts, (0, 1), value=1),  # Convert to homogenous coordinates
        )
    )
    return extrinsic(x)

# %% ../notebooks/api/00_drr.ipynb 16
class Registration(nn.Module):
    """Perform automatic 2D-to-3D registration using differentiable rendering."""

    def __init__(
        self,
        drr: DRR,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        parameterization: str,
        convention: str = None,
    ):
        super().__init__()
        self.drr = drr
        self.rotation = nn.Parameter(rotation)
        self.translation = nn.Parameter(translation)
        self.parameterization = parameterization
        self.convention = convention

    def forward(self):
        return self.drr(
            self.rotation,
            self.translation,
            parameterization=self.parameterization,
            convention=self.convention,
        )

    def get_rotation(self):
        return self.rotation.clone().detach()

    def get_translation(self):
        return self.translation.clone().detach()

    def get_pose(self):
        return convert(
            self.get_rotation(), 
            self.get_translation(), 
            parameterization=self.parameterization, 
            convention=self.convention
        )
