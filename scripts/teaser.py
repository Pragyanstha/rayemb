import os
import time
import random

import pyvista as pv
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.spatial import KDTree

from diffdrr.pose import convert
from rayemb.models import RayEmb, FixedLandmark
from rayemb.dataset import SyntheticCTPelvic1KDataset, RealDeepFluoroDataset
from rayemb.utils import setup_logger, cosine_similarity, find_batch_peak_coordinates_maxval


def run(inference, img, sampled_points, mesh, specimen_id, img_id, out_dir):
    window_size = (400, 400)
    # Build a KDTree for efficient nearest-neighbor queries
    tree = KDTree(sampled_points)

    # Choose a random starting point
    start_point = sampled_points[random.randint(0, len(sampled_points) - 1)]

    # Create a random trajectory
    trajectory = [start_point]
    num_steps = 50 - 1  # Number of steps in the trajectory

    for _ in range(num_steps):
        current_point = trajectory[-1]
        # Query nearest neighbors within a distance threshold
        neighbors = tree.query_ball_point(current_point, r=15)  # Distance threshold
        if neighbors:
            # Randomly choose the next point from the neighbors
            next_point = sampled_points[random.choice(neighbors)]
            trajectory.append(next_point)

    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)

    n_spline_points = 500  # Number of points along the spline
    full_spline = pv.Spline(trajectory, n_spline_points)

    # Inference
    sims, gt_proj_points, pred_proj_points, max_vals = inference(torch.from_numpy(full_spline.points).float().to(device))
    localization_error = np.linalg.norm(gt_proj_points - pred_proj_points, axis=-1)
    # Precompute scalars for the full spline
    precomputed_scalars = np.linspace(0, 1, n_spline_points)

    # PyVista Plotter
    trajectory_plotter = pv.Plotter(window_size=window_size)
    trajectory_plotter.add_title(f"{specimen_id}", font_size=8)
    trajectory_plotter.enable_anti_aliasing('ssaa')
    trajectory_plotter.add_mesh(mesh, opacity=0.3, color="white", label="Volume Mesh")
    camera = trajectory_plotter.camera
    L = 200
    H = 200
    camera.position = trajectory[0] + np.array([0, -L, H])
    camera.zoom(1.0)
    camera.focal_point = trajectory[0]
    # compute camera paths
    cam_positions = []
    for i in range(len(full_spline.points)+1):
        theta = 2 * np.pi * (i / len(full_spline.points))
        cam_positions.append(trajectory[0] + np.array([L * np.sin(theta), -L * np.cos(theta), H]))
    # Add an initial spline to the plotter (empty at start)
    spline_polydata = pv.PolyData(full_spline.points[:1])
    spline_polydata["scalars"] = precomputed_scalars[:1]
    spline_actor = trajectory_plotter.add_mesh(spline_polydata, 
                     scalars="scalars", 
                     cmap="magma", 
                     line_width=4, 
                     label="Trajectory")
    ball_point = pv.Sphere(radius=1.0, center=trajectory[0])
    ball_actor = trajectory_plotter.add_mesh(ball_point, color="blue", opacity=0.8, label="Start Point")
    trajectory_plotter.remove_scalar_bar()

    # Plotter 2: Heatmap
    heatmap_plotter = pv.Plotter(window_size=window_size)
    heatmap_plotter.add_title(f"Image Id : {img_id}", font_size=8)
    heatmap_plotter.enable_anti_aliasing('ssaa')
    image_grid = pv.ImageData(dimensions=(img.shape[1], img.shape[0], 1))
    image_grid.point_data["values"] = img.flatten(order="F")
    heatmap_grid = pv.ImageData(dimensions=(sims.shape[2], sims.shape[1], 1))
    heatmap_grid.point_data["values"] = sims[0].flatten(order="F")
    heatmap_plotter.add_mesh(image_grid, cmap="gray", opacity=1.0, label="Image")
    heatmap_actor = heatmap_plotter.add_mesh(
        heatmap_grid,
        cmap="jet",
        opacity=0.4,
        label="Heatmap"
    )
    # GT Projection Point is in (x, y) format
    gt_projection_ball = pv.Sphere(radius=3, center=(gt_proj_points[0][1], gt_proj_points[0][0], 3))
    gt_projection_ball_actor = heatmap_plotter.add_mesh(
        gt_projection_ball, 
        color="red",
        lighting=False, 
        opacity=0.5, 
        label="GT Projection Point"
    )
    pred_projection_ball = pv.Sphere(radius=3, center=(pred_proj_points[0][1], pred_proj_points[0][0], 3))
    pred_projection_ball_actor = heatmap_plotter.add_mesh(
        pred_projection_ball, 
        color="blue",
        lighting=False, 
        opacity=0.8, 
        label="Projection Point"
    )
    text_actor = heatmap_plotter.add_text(f"Localization Error : {localization_error[0]:.2f} px", font_size=6, position="lower_edge")
    heatmap_plotter.view_xy()
    heatmap_plotter.camera.up = (-1, 0, 0) # The image grid in on x-y plane with y-axis = i
    heatmap_plotter.remove_scalar_bar()

    # Animation callback
    def update_spline(frame):
        # Update spline points incrementally
        new_spline_points = full_spline.points[:frame]
        spline_polydata.points = new_spline_points
        if len(new_spline_points) > 1:
            # Create a valid line connectivity array
            n_segments = len(new_spline_points) - 1
            lines = []
            for i in range(n_segments):
                lines.append([2, i, i + 1])  # Each line has 2 points (i, i+1)
            lines = np.array(lines).flatten()
            spline_polydata.lines = lines
            new_ball_point = pv.Sphere(radius=0.3, center=new_spline_points[-1])
            ball_actor.mapper.SetInputData(new_ball_point)
            # Update the mapper to refresh the colors
            spline_actor.mapper.SetInputData(spline_polydata)
            camera.focal_point = (new_spline_points[-1] + trajectory[0]) / 2
            camera.position = cam_positions[frame]

        # Update heatmap in second window
        if frame < sims.shape[0]:
            heatmap_grid.point_data["values"] = sims[frame].flatten(order="F")
            heatmap_actor.mapper.SetInputData(heatmap_grid)
            new_pred_projection_ball = pv.Sphere(radius=3, center=(pred_proj_points[frame][1], pred_proj_points[frame][0], 3))
            pred_projection_ball_actor.mapper.SetInputData(new_pred_projection_ball)
            new_gt_projection_ball = pv.Sphere(radius=3, center=(gt_proj_points[frame][1], gt_proj_points[frame][0], 3))
            gt_projection_ball_actor.mapper.SetInputData(new_gt_projection_ball)
            text_actor.set_text("lower_edge", f"Localization Error : {localization_error[frame]:.2f} px")
            heatmap_plotter.render()

    # Set up animation
    trajectory_plotter.open_gif(os.path.join(out_dir, "trajectory.gif"))
    heatmap_plotter.open_gif(os.path.join(out_dir, "heatmap.gif"))
    n_frames = len(full_spline.points)

    for i in tqdm(range(2, n_frames+1)):
        update_spline(i)
        trajectory_plotter.write_frame()
        trajectory_plotter.render()
        heatmap_plotter.write_frame()
        heatmap_plotter.render()

    trajectory_plotter.close()
    heatmap_plotter.close()
if __name__ == "__main__":
    out_dir = './results/' + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    # Model parameters
    checkpoint_path = './checkpoints/rayemb-ctpelvic1k.ckpt'
    device = 'cuda'
    image_size = 224
    emb_dim = 32

    # Dataset parameters
    specimen_id = 'specimen_2'
    num_samples = 300
    h5_dir = './data/ipcai_2020_full_res_data.h5'
    template_dir = './data/deepfluoro_templates'

    model = RayEmb.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device(device),
        similarity_fn=cosine_similarity,
        image_size=image_size,
        emb_dim=emb_dim,
    )
    model.eval()

    dataset = RealDeepFluoroDataset(
        specimen_id=specimen_id,
        num_samples=num_samples,
        h5_dir=h5_dir,
        sampling_distance=1,
        template_dir=template_dir,
        sample_only_visible=True,
        image_size=image_size,
    )

    data_idx = 0
    data = dataset[data_idx]

    gt_pose = data["diffdrr_pose"]
    gt_pose = convert(gt_pose, parameterization='matrix').to(device)

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device).float()

    imgs = data["img"]
    img_id = data["img_id"]
    templates = data["templates"]
    projection_matrices = data["projection_matrices"]
    sampled_points = data["sampled_points"][:, :-1]
    img_sensor_size = data["img_sensor_size"][0].item()
    source_to_detector_distance = data["source_to_detector_distance"]
    vol_landmark = data["gt_vol_landmarks"].cpu().numpy()
    proj_landmark = data["gt_proj_landmarks"].cpu().numpy()
    query_projection_matrix = data["query_projection_matrix"]
    K = data["K"]
    diffdrr_pose = data["diffdrr_pose"]
    gt_extrinsic = data["extrinsic"]
    gt_proj_point = data["proj_point_img"].cpu().numpy()
    pixel_size = data["pixel_size"].item()
    flip_xz = data["flip_xz"]
    translate = data["translate"]
    flip_xz = convert(flip_xz, parameterization='matrix')
    translate = convert(translate, parameterization='matrix')

    def inference(sampled_points):
        sims = model.inference(imgs, templates, sampled_points, projection_matrices, return_features=False)
        pred_proj_point, max_vals = find_batch_peak_coordinates_maxval(sims)
        sims = sims ** 4 ## For better visualization
        # homogeneous coordinates
        sampled_points = torch.cat([sampled_points, torch.ones((sampled_points.shape[0], 1)).to(device)], axis=-1)
        gt_proj_point_img = query_projection_matrix @ sampled_points.T
        gt_proj_point_img = gt_proj_point_img[:2, :] / gt_proj_point_img[2, :] - 50.0
        gt_proj_point_img = gt_proj_point_img / (img_sensor_size-100) * image_size
        gt_proj_point_img = gt_proj_point_img.T # (N, 2)
        gt_proj_point_img = image_size - gt_proj_point_img
        return sims.cpu().numpy(), gt_proj_point_img.cpu().numpy(), pred_proj_point.cpu().numpy(), max_vals.cpu().numpy()

    volume = dataset.specimen.volume
    volume = np.flip(volume, axis=0)
    spacing = dataset.specimen.spacing
    sampled_points = dataset.sampled_points
    sampled_points = np.concatenate([sampled_points, np.ones((sampled_points.shape[0], 1))], axis=-1)
    sampled_points = torch.tensor(sampled_points).float().to(device)

    # Find visible points
    query_projection_matrix = gt_pose.inverse().compose(translate).compose(flip_xz)
    query_projection_matrix = K @ query_projection_matrix[0][:-1, :]
    proj_point_img = query_projection_matrix @ sampled_points.T
    proj_point_img = proj_point_img[:2, :] / proj_point_img[2, :] - 50.0
    proj_point_img = proj_point_img / (img_sensor_size-100) * image_size
    proj_point_img = proj_point_img.T # (N, 2)
    visible_indices = (proj_point_img[:, 0] >= 0) & (proj_point_img[:, 0] < image_size) & (proj_point_img[:, 1] >= 0) & (proj_point_img[:, 1] < image_size)
    visible_points = sampled_points[visible_indices]
    visible_points = visible_points.cpu().numpy()[:, :-1]

    grid = pv.ImageData(
        dimensions=volume.shape,
        spacing=spacing,
        origin=(0, 0, 0),
    )
    threshold = 300
    grid.point_data["values"] = (
        volume.flatten(order="F") > threshold
    )
    mesh = grid.contour_labeled(smoothing=True, progress_bar=True)
    run(
        inference, 
        imgs[0].detach().cpu().numpy(), 
        visible_points, 
        mesh,
        specimen_id,
        img_id,
        out_dir,
    )
