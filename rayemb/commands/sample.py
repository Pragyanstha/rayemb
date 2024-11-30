import os
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import gc
import nibabel as nib

from diffdrr.drr import DRR, Registration
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from diffdrr.pose import convert
from diffdrr.visualization import img_to_mesh

from rayemb.dataset import RayEmbCTPelvic1KDataset
from rayemb.models.rayemb_subspace import RayEmbSubspace
from rayemb.utils import cosine_similarity, NumpyEncoder
from rayemb.registration import PnPRunner, CorrespondenceSampler


matplotlib.use('TkAgg')

OUTDIR = "results"

# Filter landmarks to exclude those that are out of the image bounds
def filter_landmarks(landmarks, image_size):
    valid_indices = (landmarks[:, 0] >= 0) & (landmarks[:, 0] < image_size) & \
                    (landmarks[:, 1] >= 0) & (landmarks[:, 1] < image_size)
    return landmarks[valid_indices]

def registration():
    basefolderdir = os.path.join(OUTDIR, f"{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(basefolderdir, exist_ok=True)
    # Initialize dataset
    data_dir = './data/ctpelvic1k_synthetic_test'
    template_dir = 'data/ctpelvic1k_templates_v1'
    vol_dir = 'data/CTPelvic1K/dataset6_volume'
    mask_dir = 'data/CTPelvic1K/dataset6_label'
    # checkpoint_path = "checkpoints/lab_rayemb_dinov2_deepfluoro_cont/rayemb-14-3.68.ckpt"
    checkpoint_path = "checkpoints/pegasus_rayemb_dinov2_ctpelvic1k/rayemb-10-7.80.ckpt"
    model_name = "dinov2"
    image_size = 224
    sampling_distance = 1
    num_samples = 10000
    sample_only_visible = False
    top_k = 8000
    emb_dim = 32
    lr_rotation = 7e-3
    lr_translation = 7e0
    n_iters = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RayEmbCTPelvic1KDataset(
        image_size=image_size,
        data_dir=data_dir,
        vol_dir=vol_dir,
        mask_dir=mask_dir,
        num_samples=num_samples,
        template_dir=template_dir,
        split='test', 
        sample_only_visible=False,
        sampling_distance=sampling_distance,
        augment=False)

    # Initialize the model
    model = RayEmbSubspace.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
        similarity_fn=cosine_similarity,
        emb_dim=emb_dim,
        model_name=model_name,
        image_size=image_size)
    model.eval()
    
    correspondence_sampler = CorrespondenceSampler(model, top_k=top_k)
    pnp_runner = PnPRunner(dataset, correspondence_sampler=correspondence_sampler, device=device)

    for data_idx in range(len(dataset)):
        print(f"Processing {data_idx}")
        data = dataset[data_idx]
        specimen_id = data["specimen_id"]
        nifti_vol = nib.load(f"{vol_dir}/{specimen_id}.nii.gz")
        volume = nifti_vol.get_fdata()
        voxel_spacings = np.array(nifti_vol.header.get_zooms())
        _, pred_pose, viz = pnp_runner(data_idx, visualize=True, off_screen=True, volume=volume, voxel_spacings=voxel_spacings)

        img_target = data["img"][0].to(device).double()[None, None, ...]
        img_sensor_size = int(data["img_sensor_size"][0].item())
        gt_pose = data["diffdrr_pose"]
        gt_pose = convert(gt_pose, parameterization='matrix').to(device)
        gt_vol_landmarks = data["gt_vol_landmarks"].unsqueeze(0).to(device)
        img_id = data["img_id"]
        pixel_size = data["pixel_size"].item()
        flip_xz = convert(data["flip_xz"].to(device), parameterization='matrix')
        translate = convert(data["translate"].to(device), parameterization='matrix')

        folderdir = os.path.join(basefolderdir, f"{specimen_id}", f"{img_id}")
        os.makedirs(folderdir, exist_ok=True)

        drr_moving = DRR(
            volume,
            voxel_spacings,
            data["source_to_detector_distance"].item() / 2.0,
            image_size,
            data["pixel_size"].item() * img_sensor_size / image_size,
            x0=0.0,
            y0=0.0,
            reverse_x_axis=True,
            renderer="siddon",
            bone_attenuation_multiplier=2.5
        ).to(device)

        rotation, translation = pred_pose.convert(parameterization='se3_log_map')
        drr_registration = Registration(drr_moving, rotation=rotation, translation=translation, parameterization='se3_log_map').to(device)

        optimizer = torch.optim.Adam(
                [
                    {"params": [drr_registration.rotation], "lr": lr_rotation},
                    {"params": [drr_registration.translation], "lr": lr_translation},
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
        criterion = MultiscaleNormalizedCrossCorrelation2d(patch_sizes, patch_weights).to(device).double()

        p = viz["plotter"]
        c_act = viz["c_act"]
        d_act = viz["d_act"]
        p_act = viz["p_act"]
        best_loss = 0
        optim_pose = pred_pose
        for i in range(n_iters):
            optimizer.zero_grad()
            pred_drr = drr_registration()
            loss = criterion(pred_drr, img_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss.item() > best_loss:
                best_loss = loss.item()
                optim_pose = drr_registration.get_pose()

            print(f"Iter {i}: {loss.item()}")

        camera, detector, texture, principal_ray = img_to_mesh(drr_moving, optim_pose)
        c_act = p.add_mesh(camera, color='b')
        d_act = p.add_mesh(detector, texture=texture)
        p_act = p.add_mesh(principal_ray, color='b')
        p.save_graphic(os.path.join(folderdir, f"registration.svg"))
        p.close()

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

        print(f"Init reprojection error: {init_reprojection_error_detector_mean}")
        print(f"Optim reprojection error: {optim_reprojection_error_detector_mean}")

        print(f"Init landmarks cam error: {init_landmarks_cam_error.mean()}")
        print(f"Optim landmarks cam error: {optim_landmarks_cam_error.mean()}")

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
        plt.savefig(os.path.join(folderdir, f"projections.png"))
        plt.close(fig)

        data = {
            "meta": {
                "specimen_id": data["specimen_id"],
                "img_id": img_id,
                "pixel_size": data["pixel_size"].item(),
                "img_sensor_size": img_sensor_size,
                "image_size": image_size,
                "parameters": {
                    "sampling_distance": sampling_distance,
                    "num_samples": num_samples,
                    "dataset": "DFL_synthetic_v4",
                    "model": "rayemb_subspace",
                    "emb_dim": 32,
                    "sample_only_visible": sample_only_visible,
                    "lr_rotation": lr_rotation,
                    "lr_translation": lr_translation,
                    "n_iters_diffdrr": n_iters,
                    "top_k": top_k,
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
        with open(os.path.join(folderdir, f"camera.json"), "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

        del drr_registration, drr_moving
        del img_target, pred_drr, gt_drr, init_drr  # Delete large tensors
        del viz, p, c_act, d_act, p_act
        # Ensure all local variables that are large are deleted
        del data, pred_pose, optim_pose, gt_pose, gt_vol_landmarks
        torch.cuda.empty_cache()  # Clear unused memory
        gc.collect()

if __name__ == "__main__":
    registration()
