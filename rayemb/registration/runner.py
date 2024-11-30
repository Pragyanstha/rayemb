from typing import Any
import logging
import torch
import numpy as np
import cv2
import kornia as kn
from pyvista.plotting.plotter import Plotter

from diffdrr.drr import DRR
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from diffdrr.pose import RigidTransform, convert
from rayemb.utils import setup_logger

logger = setup_logger(__name__)

class PnPRunner():
    def __init__(self, correspondence_sampler, device, is_real=False):
        self.correspondence_sampler = correspondence_sampler
        self.device = device
        self.inlier_threshold = 10.0
        self.num_iterations = 10000
        self.confidence = 0.9999
        self.method = cv2.USAC_MAGSAC
        self.is_real = is_real

    def compute_extrinsic(self, r_vec, t_vec):
        t = torch.tensor(t_vec).float()
        R = kn.geometry.conversions.axis_angle_to_rotation_matrix(torch.from_numpy(r_vec).view(1, 3)).float()[0]
        P = torch.cat((R, t), dim=-1)
        return torch.cat((P, torch.tensor([0, 0, 0, 1]).float().view(1, 4)), dim=0)

    def __call__(self, data, visualize=False, off_screen=False, voxel_spacings=None, volume=None, specimen=None) -> Any:
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device).float()

        specimen_id = data["specimen_id"]
        img_id = data["img_id"]
        imgs = data["img"]
        templates = data["templates"]
        projection_matrices = data["projection_matrices"]
        sampled_points = data["sampled_points"][:, :-1]
        img_sensor_size = data["img_sensor_size"][0].item()
        source_to_detector_distance = data["source_to_detector_distance"]
        specimen_id = data["specimen_id"]
        K = data["K"].cpu().numpy()
        diffdrr_pose = data["diffdrr_pose"]
        gt_proj_point = data["proj_point_img"].cpu().numpy()
        pixel_size = data["pixel_size"].item()
        image_size = imgs.shape[-1] # Square image size
        sampled_points_np, proj_points_np, max_vals, sims, sampled_indices = self.correspondence_sampler(imgs, templates, sampled_points, projection_matrices)
        gt_proj_point = gt_proj_point[sampled_indices] # only keep the ones that are in the sampled points

        # This is for flipping the signs of intrinsic matrix
        if self.is_real:
            proj_points_np = proj_points_np / image_size * (img_sensor_size - 100)
            proj_points_np += 50
        else:
            proj_points_np = proj_points_np / image_size * img_sensor_size
        proj_points_np[:, 1] = img_sensor_size - proj_points_np[:, 1]
        dist_coeffs = np.zeros(4)
        success, rotation_vector, translation_vector, inliers  = cv2.solvePnPRansac(
            sampled_points_np, proj_points_np, K, dist_coeffs,
            iterationsCount=self.num_iterations,
            reprojectionError=self.inlier_threshold,
            confidence=self.confidence,
            flags=self.method
        )

        pred_extrinsic = self.compute_extrinsic(rotation_vector, translation_vector)

        flip_xz = RigidTransform(
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).to(self.device)
        translate = RigidTransform(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, -source_to_detector_distance.item() / 2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).to(self.device)

        pred_pose = convert(pred_extrinsic, parameterization="matrix").to(self.device)
        pred_pose = translate.compose(flip_xz).compose(pred_pose.inverse())

        if visualize:
            gt_pose = convert(diffdrr_pose[None ,...], parameterization="matrix").to(self.device)
            drr = DRR(
                specimen.volume if specimen else volume,
                specimen.spacing if specimen else voxel_spacings,
                source_to_detector_distance.item() / 2.0,
                image_size,
                pixel_size * (img_sensor_size - 100.0) / image_size if self.is_real else pixel_size * img_sensor_size / image_size,
                x0=specimen.x0 if specimen else 0.0,
                y0=specimen.y0 if specimen else 0.0,
                reverse_x_axis=True,
            ).to(self.device)

            mesh = drr_to_mesh(drr, method="surface_nets")
            p = Plotter(off_screen=off_screen)
            p.add_mesh(mesh)

            camera, detector, texture, principal_ray = img_to_mesh(drr, pred_pose)
            c_act = p.add_mesh(camera, color="g")
            d_act = p.add_mesh(detector, texture=texture)
            p_act = p.add_mesh(principal_ray, color="g")

            camera, detector, texture, principal_ray = img_to_mesh(drr, gt_pose)
            p.add_mesh(camera, color="r", opacity=0.3)
            p.add_mesh(detector, texture=texture, opacity=0.5)
            p.add_mesh(principal_ray, color="r", opacity=0.5)
            p.add_title(f"{specimen_id} - {img_id}")

            return pred_extrinsic, pred_pose, {
                "c_act": c_act,
                "d_act": d_act,
                "p_act": p_act,
                "plotter": p
            }

        return pred_extrinsic, pred_pose
