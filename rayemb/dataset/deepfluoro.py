# This code was borrowed from the DiffPose project.

# %% auto 0
__all__ = ['DeepFluoroDataset', 'convert_deepfluoro_to_diffdrr', 'convert_diffdrr_to_deepfluoro', 'Evaluator', 'preprocess']

# %% ../notebooks/api/00_deepfluoro.ipynb 3
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from beartype import beartype
from diffdrr.pose import RigidTransform, convert

from .calibration import perspective_projection

# %% ../notebooks/api/00_deepfluoro.ipynb 5
@beartype
class DeepFluoroDataset(torch.utils.data.Dataset):
    """
    Get X-ray projections and poses from specimens in the `DeepFluoro` dataset.

    Given a specimen ID and projection index, returns the projection and the camera matrix for DiffDRR.
    """

    def __init__(
        self,
        id_number: int,  # Specimen number (1-6)
        filename: Optional[Union[str, Path]] = None,  # Path to DeepFluoro h5 file
        preprocess: bool = True,  # Preprocess X-rays
    ):
        # Load the volume
        (
            self.specimen,
            self.projections,
            self.volume,
            self.spacing,
            self.lps2volume,
            self.intrinsic,
            self.extrinsic,
            self.focal_len,
            self.x0,
            self.y0,
            self.num_rows,
            self.num_cols,
            self.proj_row_spacing,
            self.proj_col_spacing,
            self.mask
        ) = load_deepfluoro_dataset(id_number, filename)

        self.preprocess = preprocess
        # Get the isocenter pose (AP viewing angle at volume isocenter)
        isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
        isocenter_xyz = torch.tensor(self.volume.shape) * self.spacing / 2
        isocenter_xyz = isocenter_xyz.unsqueeze(0)
        self.isocenter_pose = convert(
            isocenter_rot.float(),
            isocenter_xyz.float(),
            parameterization="euler_angles",
            convention="ZYX",
        )

        # Camera matrices and fiducials for the specimen
        self.fiducials_deepdrr = get_3d_fiducials(self.specimen)
        self.fiducials = self.lps2volume.inverse()(self.fiducials_deepdrr)

        # Miscellaneous transformation matrices for wrangling SE(3) poses
        self.flip_xz = RigidTransform(
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        self.translate = RigidTransform(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, -self.focal_len / 2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        self.flip_180 = RigidTransform(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )


    def __len__(self):
        return len(self.projections)

    def __iter__(self):
        return iter(self[idx] for idx in range(len(self)))

    def __getitem__(self, idx):
        """
        (1) Swap the x- and z-axes
        (2) Reverse the x-axis to make the matrix E(3) -> SE(3)
        (3) Move the camera to the origin
        (4) Rotate the detector plane by 180, if offset
        (5) Form the full SE(3) transformation matrix
        """
        projection = self.projections[f"{idx:03d}"]
        img = torch.from_numpy(projection["image/pixels"][:])
        world2volume = torch.from_numpy(projection["gt-poses/cam-to-pelvis-vol"][:])
        world2volume = RigidTransform(world2volume)
        pose = convert_deepfluoro_to_diffdrr(self, world2volume)

        gt_landmarks = []
        for landmarkname in projection['gt-landmarks']:
            landmark = projection['gt-landmarks'][landmarkname][:]
            gt_landmarks.append(torch.from_numpy(landmark))
        gt_landmarks = torch.stack(gt_landmarks, dim=0).squeeze(-1)

        # Handle rotations in the imaging dataset
        if self._rot_180_for_up(idx):
            img = torch.rot90(img, k=2)
            pose = self.flip_180.compose(pose)

        # Optionally, preprocess the images
        img = img.unsqueeze(0).unsqueeze(0)
        if self.preprocess:
            img = preprocess(img)

        return img, pose, gt_landmarks

    def get_2d_projection(self, sample_3d, pose):
        # Get the fiducials from the true camera pose
        extrinsic = (
            pose.inverse().compose(self.translate).compose(self.flip_xz)
        ).to(sample_3d.device)
        projected = perspective_projection(
            extrinsic, self.intrinsic.to(sample_3d.device), sample_3d.unsqueeze(0)
        )


        if self.preprocess:
            projected -= 50

        return projected

    def get_2d_fiducials(self, idx, pose):
        # Get the fiducials from the true camera pose
        _, true_pose = self.__getitem__(idx)
        extrinsic = (
            self.lps2volume.inverse()
            .compose(true_pose.inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials_deepdrr
        )

        # Get the fiducials from the predicted camera pose
        extrinsic = (
            self.lps2volume.inverse()
            .compose(pose.cpu().inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        pred_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials_deepdrr
        )

        if self.preprocess:
            true_fiducials -= 50
            pred_fiducials -= 50

        return true_fiducials, pred_fiducials

    def _rot_180_for_up(self, idx):
        return self.projections[f"{idx:03d}"]["rot-180-for-up"][()]

# %% ../notebooks/api/00_deepfluoro.ipynb 6
def convert_deepfluoro_to_diffdrr(specimen, pose: RigidTransform):
    """Transform the camera coordinate system used in DeepFluoro to the convention used by DiffDRR."""
    return (
        specimen.translate.compose(specimen.flip_xz)
        .compose(specimen.extrinsic.inverse())
        .compose(pose)
        .compose(specimen.lps2volume.inverse())
    )


def convert_diffdrr_to_deepfluoro(specimen, pose: RigidTransform):
    """Transform the camera coordinate system used in DiffDRR to the convention used by DeepFluoro."""
    return (
        specimen.lps2volume.inverse()
        .compose(pose.inverse())
        .compose(specimen.translate)
        .compose(specimen.flip_xz)
    )

# %% ../notebooks/api/00_deepfluoro.ipynb 7
from torch.nn.functional import pad

from .calibration import perspective_projection


class Evaluator:
    def __init__(self, specimen, idx, intrinsic=None):
        # Save matrices to device
        self.translate = specimen.translate
        self.flip_xz = specimen.flip_xz
        self.intrinsic = specimen.intrinsic if intrinsic is None else intrinsic
        self.intrinsic_inv = specimen.intrinsic.inverse()

        # Get gt fiducial locations
        self.specimen = specimen
        self.fiducials = specimen.fiducials_deepdrr
        gt_pose = specimen[idx][1]
        self.true_projected_fiducials = self.project(gt_pose)

    def project(self, pose):
        extrinsic = convert_diffdrr_to_deepfluoro(self.specimen, pose)
        x = perspective_projection(extrinsic, self.intrinsic, self.fiducials)
        x = -self.specimen.focal_len * torch.einsum(
            "ij, bnj -> bni",
            self.intrinsic_inv,
            pad(x, (0, 1), value=1),  # Convert to homogenous coordinates
        )
        extrinsic = (
            self.flip_xz.inverse().compose(self.translate.inverse()).compose(pose)
        )
        return extrinsic(x)

    def __call__(self, pose):
        pred_projected_fiducials = self.project(pose)
        registration_error = (
            (self.true_projected_fiducials - pred_projected_fiducials)
            .norm(dim=-1)
            .mean()
        )
        registration_error *= 0.194  # Pixel spacing is 0.194 mm / pixel isotropic
        return registration_error

# %% ../notebooks/api/00_deepfluoro.ipynb 8
from diffdrr.utils import parse_intrinsic_matrix


def load_deepfluoro_dataset(id_number, filename):
    # Open the H5 file for the dataset
    if filename is None:
        root = Path(__file__).parent.parent.absolute()
        filename = root / "data/ipcai_2020_full_res_data.h5"
    f = h5py.File(filename, "r")
    (
        intrinsic,
        extrinsic,
        num_cols,
        num_rows,
        proj_col_spacing,
        proj_row_spacing,
    ) = parse_proj_params(f)
    focal_len, x0, y0 = parse_intrinsic_matrix(
        intrinsic,
        num_rows,
        num_cols,
        proj_row_spacing,
        proj_col_spacing,
    )

    # Try to load the particular specimen
    assert id_number in {1, 2, 3, 4, 5, 6}
    specimen_id = [
        "17-1882",
        "18-1109",
        "18-0725",
        "18-2799",
        "18-2800",
        "17-1905",
    ][id_number - 1]
    specimen = f[specimen_id]
    projections = specimen["projections"]

    # Parse the volume
    volume, spacing, lps2volume, mask = parse_volume(specimen)
    return (
        specimen,
        projections,
        volume,
        spacing,
        lps2volume,
        intrinsic,
        extrinsic,
        focal_len,
        x0,
        y0,
        num_rows,
        num_cols,
        proj_row_spacing,
        proj_col_spacing,
        mask
    )


def parse_volume(specimen):
    # Parse the volume
    spacing = specimen["vol/spacing"][:].flatten().astype(np.float32)
    volume = specimen["vol/pixels"][:].astype(np.float32)
    volume = np.swapaxes(volume, 0, 2)[::-1].copy()

    seg = specimen["vol-seg/image/pixels"][:].astype(np.float32)
    seg = np.swapaxes(seg, 0, 2).copy()
    mask = seg > 0

    # volume = volume*mask.astype(np.float32)
    # Parse the translation matrix from anatomical coordinates to world coordinates
    origin = torch.from_numpy(specimen["vol/origin"][:].flatten()).to(torch.float32)
    lps2volume = RigidTransform(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, origin[0]],
                [0.0, 1.0, 0.0, origin[1]],
                [0.0, 0.0, 1.0, origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    return volume, spacing, lps2volume, mask


def parse_proj_params(f):
    proj_params = f["proj-params"]
    extrinsic = torch.from_numpy(proj_params["extrinsic"][:])
    extrinsic = RigidTransform(extrinsic)
    intrinsic = torch.from_numpy(proj_params["intrinsic"][:])
    num_cols = float(proj_params["num-cols"][()])
    num_rows = float(proj_params["num-rows"][()])
    proj_col_spacing = float(proj_params["pixel-col-spacing"][()])
    proj_row_spacing = float(proj_params["pixel-row-spacing"][()])
    return intrinsic, extrinsic, num_cols, num_rows, proj_col_spacing, proj_row_spacing


def get_3d_fiducials(specimen):
    fiducials = []
    for landmark in specimen["vol-landmarks"]:
        pt_3d = specimen["vol-landmarks"][landmark][:]
        pt_3d = torch.from_numpy(pt_3d)
        fiducials.append(pt_3d)
    return torch.stack(fiducials, dim=0).permute(2, 0, 1) # (1, 14, 3)

# %% ../notebooks/api/00_deepfluoro.ipynb 9
from torchvision.transforms.functional import center_crop, gaussian_blur


def preprocess(img, size=None, initial_energy=torch.tensor(65487.0)):
    """
    Recover the line integral: $L[i,j] = \log I_0 - \log I_f[i,j]$

    (1) Remove edge due to collimator
    (2) Smooth the image to make less noisy
    (3) Subtract the log initial energy for each ray
    (4) Recover the line integral image
    (5) Rescale image to [0, 1]
    """
    img = center_crop(img, (1436, 1436))
    img = gaussian_blur(img, (5, 5), sigma=1.0)
    img = initial_energy.log() - img.log()
    img = (img - img.min()) / (img.max() - img.min())
    return img
