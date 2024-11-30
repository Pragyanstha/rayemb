from enum import Enum
import gc

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from diffdrr.drr import DRR, Registration
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from diffdrr.pose import convert
from diffdrr.visualization import img_to_mesh
from rayemb.utils import setup_logger, filter_landmarks
from rayemb.constants import TOP_K, N_ITERS, LR_ROTATION, LR_TRANSLATION

logger = setup_logger(__name__)

class DatasetType(Enum):
    SYNTHETIC_DEEPFLUORO = "synthetic_deepfluoro"
    SYNTHETIC_CTPELIC1K = "synthetic_ctpelvic1k"
    REAL_DEEPFLUORO = "real_deepfluoro"

class Evaluator():
    def __init__(self, model, pnp_runner, device):
        self.model = model
        self.pnp_runner = pnp_runner
        self.device = device

    def visualize_registration(self, viz, drr_moving, img_target, gt_pose, optim_pose):
        p = viz["plotter"]
        c_act = viz["c_act"]
        d_act = viz["d_act"]
        p_act = viz["p_act"]        
        camera, detector, texture, principal_ray = img_to_mesh(drr_moving, optim_pose)
        c_act = p.add_mesh(camera, color='b')
        d_act = p.add_mesh(detector, texture=texture)
        p_act = p.add_mesh(principal_ray, color='b')
        return p

    def visualize_projections(self, 
                             img_target, 
                             init_landmarks_filtered,
                             optim_landmarks_filtered,
                             gt_landmarks_filtered,
                             init_drr,
                             pred_drr,
                             gt_drr,
                             init_reprojection_error_detector_mean,
                             optim_reprojection_error_detector_mean
                             ):
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(2, 4, 1)
        plt.imshow(img_target, cmap="gray")
        plt.plot(init_landmarks_filtered[:, 0], init_landmarks_filtered[:, 1], 'x', color='g', markersize=5)
        plt.plot(optim_landmarks_filtered[:, 0], optim_landmarks_filtered[:, 1], 'x', color='b', markersize=5)
        plt.plot(gt_landmarks_filtered[:, 0], gt_landmarks_filtered[:, 1], 'x', color='r', markersize=5)
        plt.axis('off')
        plt.title("Target")
        plt.subplot(2, 4, 2)
        plt.imshow(init_drr, cmap="gray")
        plt.plot(init_landmarks_filtered[:, 0], init_landmarks_filtered[:, 1], 'x', color='g', markersize=5)
        plt.title(f"Initial (PnP-Reg) {init_reprojection_error_detector_mean:.2f}mm")
        plt.axis('off')
        plt.subplot(2, 4, 3)
        plt.imshow(pred_drr, cmap="gray")
        plt.plot(optim_landmarks_filtered[:, 0], optim_landmarks_filtered[:, 1], 'x', color='b', markersize=5)
        plt.title(f"Optimized (PnP-Reg + DiffDRR) {optim_reprojection_error_detector_mean:.2f}mm")
        plt.axis('off')
        plt.subplot(2, 4, 4)
        plt.imshow(gt_drr, cmap="gray")
        plt.plot(gt_landmarks_filtered[:, 0], gt_landmarks_filtered[:, 1], 'x', color='r', markersize=5)
        plt.title("GT Rerendered")
        plt.axis('off')
        plt.subplot(2, 4, 5)
        plt.imshow(np.abs(init_drr - pred_drr), cmap="inferno", vmin=0.0, vmax=1.0)
        plt.title("Initial-Optimized Error")
        plt.axis('off')
        plt.subplot(2, 4, 6)
        plt.imshow(np.abs(init_drr - img_target), cmap="inferno", vmin=0.0, vmax=1.0)
        plt.title("Initial-Target Error")
        plt.axis('off')
        plt.subplot(2, 4, 7)
        plt.imshow(np.abs(pred_drr - img_target), cmap='inferno', vmin=0.0, vmax=1.0)
        plt.title("Optimized-Target Error")
        plt.axis('off')
        plt.subplot(2, 4, 8)
        plt.imshow(np.abs(gt_drr - img_target), cmap="inferno", vmin=0.0, vmax=1.0)
        plt.title("GT-Target Error")
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5) 
        return fig

    def __call__(self, data, volume=None, voxel_spacings=None, specimen=None, visualize=True, off_screen=True):
        specimen_id = data["specimen_id"]
        if (volume is None and specimen is None) or (volume is not None and specimen is not None):
            raise ValueError("Either volume or specimen must be provided")
        
        _, pred_pose, viz = self.pnp_runner(data, visualize=visualize, off_screen=off_screen, volume=volume, voxel_spacings=voxel_spacings, specimen=specimen)
        img_target = data["img"][0].to(self.device).double()[None, None, ...]
        img_sensor_size = int(data["img_sensor_size"][0].item())
        image_size = img_target.shape[-1] # Assumes square image
        gt_pose = data["diffdrr_pose"]
        gt_pose = convert(gt_pose, parameterization='matrix').to(self.device)
        gt_vol_landmarks = data["gt_vol_landmarks"].unsqueeze(0).to(self.device)
        img_id = data["img_id"]
        pixel_size = data["pixel_size"].item()
        flip_xz = convert(data["flip_xz"].to(self.device), parameterization='matrix')
        translate = convert(data["translate"].to(self.device), parameterization='matrix')

        drr_moving = DRR(
            volume if volume is not None else specimen.volume.astype(np.float64),
            voxel_spacings if voxel_spacings is not None else specimen.spacing.astype(np.float64),
            data["source_to_detector_distance"].item() / 2.0,
            image_size,
            data["pixel_size"].item() * img_sensor_size / image_size if volume is not None else data["pixel_size"].item() * (img_sensor_size - 100.0) / image_size,
            x0=specimen.x0 if specimen is not None else 0.0,
            y0=specimen.y0 if specimen is not None else 0.0,
            reverse_x_axis=True,
            renderer="siddon",
            bone_attenuation_multiplier=2.5
        ).to(self.device)

        rotation, translation = pred_pose.convert(parameterization='se3_log_map')
        drr_registration = Registration(drr_moving, rotation=rotation, translation=translation, parameterization='se3_log_map').to(self.device)

        optimizer = torch.optim.Adam(
                [
                    {"params": [drr_registration.rotation], "lr": LR_ROTATION},
                    {"params": [drr_registration.translation], "lr": LR_TRANSLATION},
                ],
                maximize=True,
                amsgrad=True,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.9,
        )
        patch_sizes = [None, 23]  # Define the patch sizes for different scales
        patch_weights = [0.5, 0.5]  # Weights for each scale
        criterion = MultiscaleNormalizedCrossCorrelation2d(patch_sizes, patch_weights).to(self.device).double()

        best_loss = 0
        optim_pose = pred_pose
        for i in range(N_ITERS):
            optimizer.zero_grad()
            pred_drr = drr_registration()
            loss = criterion(pred_drr, img_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss.item() > best_loss:
                best_loss = loss.item()
                optim_pose = drr_registration.get_pose()

            logger.info(f"Iter {i}: {loss.item()}")

        gt_drr = drr_moving(gt_pose)
        gt_drr = gt_drr[0].detach().cpu().permute(1, 2, 0).numpy()
        pred_drr = drr_moving(optim_pose)
        pred_drr = pred_drr[0].detach().cpu().permute(1, 2, 0).numpy()
        init_drr = drr_moving(pred_pose)
        init_drr = init_drr[0].detach().cpu().permute(1, 2, 0).numpy()

        img_target = img_target[0].cpu().permute(1, 2, 0).numpy()
        # normalize pred_drr and gt_drr
        init_drr = (init_drr - init_drr.min()) / (init_drr.max() - init_drr.min())
        pred_drr = (pred_drr - pred_drr.min()) / (pred_drr.max() - pred_drr.min())
        gt_drr = (gt_drr - gt_drr.min()) / (gt_drr.max() - gt_drr.min())

        init_landmarks = drr_moving.perspective_projection(pred_pose, gt_vol_landmarks).cpu().numpy()[0]
        optim_landmarks = drr_moving.perspective_projection(optim_pose, gt_vol_landmarks).cpu().numpy()[0]
        gt_landmarks = drr_moving.perspective_projection(gt_pose, gt_vol_landmarks).cpu().numpy()[0]

        init_reprojection_error_pixels = np.linalg.norm(init_landmarks - gt_landmarks, axis=-1)
        init_reprojection_error_detector_mean = init_reprojection_error_pixels.mean() * pixel_size * (img_sensor_size) / image_size

        optim_reprojection_error_pixels = np.linalg.norm(optim_landmarks - gt_landmarks, axis=-1)
        optim_reprojection_error_detector_mean = optim_reprojection_error_pixels.mean() * pixel_size * (img_sensor_size) / image_size
        
        gt_extrinsic = (
            gt_pose.inverse().compose(translate).compose(flip_xz)
        )

        pred_extrinsic = (
            pred_pose.inverse().compose(translate).compose(flip_xz)
        )

        optim_extrinsic = (
            optim_pose.inverse().compose(translate).compose(flip_xz)
        )

        pred_vol_landmarks_cam = pred_extrinsic(gt_vol_landmarks)[0].cpu().numpy()
        gt_vol_landmarks_cam = gt_extrinsic(gt_vol_landmarks)[0].cpu().numpy()
        optim_vol_landmarks_cam = optim_extrinsic(gt_vol_landmarks)[0].cpu().numpy()

        init_landmarks_cam_error = np.linalg.norm(pred_vol_landmarks_cam - gt_vol_landmarks_cam, axis=-1)
        optim_landmarks_cam_error = np.linalg.norm(optim_vol_landmarks_cam - gt_vol_landmarks_cam, axis=-1)

        init_landmarks_filtered = filter_landmarks(init_landmarks, image_size)
        optim_landmarks_filtered = filter_landmarks(optim_landmarks, image_size)
        gt_landmarks_filtered = filter_landmarks(gt_landmarks, image_size)

        logger.info(f"Init reprojection error: {init_reprojection_error_detector_mean}")
        logger.info(f"Optim reprojection error: {optim_reprojection_error_detector_mean}")

        logger.info(f"Init landmarks cam error: {init_landmarks_cam_error.mean()}")
        logger.info(f"Optim landmarks cam error: {optim_landmarks_cam_error.mean()}")

        p = self.visualize_registration(viz, drr_moving, img_target, gt_pose, optim_pose)
        fig = self.visualize_projections(img_target,
                                        init_landmarks_filtered, 
                                        optim_landmarks_filtered, 
                                        gt_landmarks_filtered, 
                                        init_drr, 
                                        pred_drr, 
                                        gt_drr, 
                                        init_reprojection_error_detector_mean, 
                                        optim_reprojection_error_detector_mean)
        data = {
            "meta": {
                "specimen_id": data["specimen_id"],
                "img_id": img_id,
                "pixel_size": data["pixel_size"].item(),
                "img_sensor_size": img_sensor_size,
                "image_size": image_size,
                "parameters": {
                    "model": self.model.__class__.__name__,
                    "emb_dim": self.model.emb_dim,
                    "lr_rotation": LR_ROTATION,
                    "lr_translation": LR_TRANSLATION,
                    "n_iters_diffdrr": N_ITERS,
                    "top_k": TOP_K,
                }
            },
            "gt_pose": gt_pose.matrix.cpu().numpy().tolist(),
            "initial_pose": pred_pose.matrix.cpu().numpy().tolist(),
            "optim_pose": optim_pose.matrix.cpu().numpy().tolist(),
            "pred_landmarks": optim_landmarks.tolist(),
            "gt_landmarks": gt_landmarks.tolist(),
            "init_reprojection_error_pixel": init_reprojection_error_pixels.tolist(),
            "init_reprojection_error_detector_mean": init_reprojection_error_detector_mean,
            "init_landmarks_cam_error": init_landmarks_cam_error.tolist(),
            "init_landmarks_cam_error_mean": init_landmarks_cam_error.mean(),
            "optim_reprojection_error_pixel": optim_reprojection_error_pixels.tolist(),
            "optim_reprojection_error_detector_mean": optim_reprojection_error_detector_mean,
            "optim_landmarks_cam_error": optim_landmarks_cam_error.tolist(),
            "optim_landmarks_cam_error_mean": optim_landmarks_cam_error.mean(),
        }
        del drr_registration, drr_moving
        del img_target, pred_drr, gt_drr, init_drr  # Delete large tensors
        del  pred_pose, optim_pose, gt_pose, gt_vol_landmarks
        torch.cuda.empty_cache()  # Clear unused memory
        gc.collect()

        return data, p, fig
