from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from beartype import beartype
import torch

def load(id_number, height, device):
    from rayemb.dataset.deepfluoro import DeepFluoroDataset
    specimen = DeepFluoroDataset(id_number, filename="data/ipcai_2020_full_res_data.h5", preprocess=True)
    isocenter_pose = specimen.isocenter_pose.to(device)

    delx = 1.4
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        1800 / 2.0,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        renderer="siddon",
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr

from torchvision.transforms import Compose, Lambda, Normalize, Resize


@beartype
def get_random_offset_params(batch_size: int, params: dict, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, params["r1"]).sample((batch_size,))
    r2 = torch.distributions.Normal(0, params["r2"]).sample((batch_size,))
    r3 = torch.distributions.Normal(0, params["r3"]).sample((batch_size,))
    t1 = torch.distributions.Normal(params["t1"][0], params["t1"][1]).sample((batch_size,))
    t2 = torch.distributions.Normal(params["t2"][0], params["t2"][1]).sample((batch_size,))
    t3 = torch.distributions.Normal(params["t3"][0], params["t3"][1]).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        log_R_vee,
        log_t_vee,
        parameterization="se3_log_map",
    )

@beartype
def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    t1 = torch.distributions.Normal(10, 70).sample((batch_size,))
    t2 = torch.distributions.Normal(250, 90).sample((batch_size,))
    t3 = torch.distributions.Normal(5, 50).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        log_R_vee,
        log_t_vee,
        parameterization="se3_log_map",
    )


class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.transforms = Compose(
            [
                Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)