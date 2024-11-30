# This code was borrowed from the DiffPose project.

# %% auto 0
__all__ = ['perspective_projection']

# %% ../notebooks/api/02_calibration.ipynb 4
import torch

# %% ../notebooks/api/02_calibration.ipynb 6
from typing import Optional

from beartype import beartype
from diffdrr.pose import RigidTransform
from jaxtyping import Float, jaxtyped

# %% ../notebooks/api/02_calibration.ipynb 7
@jaxtyped(typechecker=beartype)
def perspective_projection(
    extrinsic: RigidTransform,  # Extrinsic camera matrix (world to camera)
    intrinsic: Float[torch.Tensor, "3 3"],  # Intrinsic camera matrix (camera to image)
    x: Float[torch.Tensor, "b n 3"],  # World coordinates
) -> Float[torch.Tensor, "b n 2"]:
    x = extrinsic(x)
    x = torch.einsum("ij, bnj -> bni", intrinsic, x)
    z = x[..., -1].unsqueeze(-1).clone()
    x = x / z
    return x[..., :2]