import numpy as np
import pyvista as pv
from scipy import ndimage
import torch


def sample_grid_volume_from_ct(ct_volume, spacing, threshold=350, resolution=15):
    # Apply threshold to identify bone structure
    ct_volume = np.flip(ct_volume, axis=0)
    bone_mask = ct_volume >= threshold
    
    # Get the bounds of the CT volume
    x_bounds = (0, ct_volume.shape[0])
    y_bounds = (0, ct_volume.shape[1])
    z_bounds = (0, ct_volume.shape[2])

    # Generate grid points within the bounding box
    x = np.arange(x_bounds[0], x_bounds[1], resolution)
    y = np.arange(y_bounds[0], y_bounds[1], resolution)
    z = np.arange(z_bounds[0], z_bounds[1], resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Convert grid points to integer indices
    grid_indices = grid_points.astype(int)
    
    # Ensure indices are within the bounds of the CT volume
    valid_indices = (
        (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < ct_volume.shape[0]) &
        (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < ct_volume.shape[1]) &
        (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < ct_volume.shape[2])
    )
    
    # Filter grid points to include only those within the bone mask
    grid_indices = grid_indices[valid_indices]
    selected_mask = bone_mask[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    selected_grid_points = grid_points[valid_indices][selected_mask]

    return selected_grid_points * np.array(spacing)

def sample_grid_from_mask(mask, spacing, resolution=15):
    # Fill holes in the mask
    mask = ndimage.binary_fill_holes(mask)
    # Get the bounds of the CT volume
    x_bounds = (0, mask.shape[0])
    y_bounds = (0, mask.shape[1])
    z_bounds = (0, mask.shape[2])

    # Generate grid points within the bounding box
    x = np.arange(x_bounds[0], x_bounds[1], resolution)
    y = np.arange(y_bounds[0], y_bounds[1], resolution)
    z = np.arange(z_bounds[0], z_bounds[1], resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Convert grid points to integer indices
    grid_indices = grid_points.astype(int)
    
    # Ensure indices are within the bounds of the CT volume
    valid_indices = (
        (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < mask.shape[0]) &
        (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < mask.shape[1]) &
        (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < mask.shape[2])
    )
    
    # Filter grid points to include only those within the bone mask
    grid_indices = grid_indices[valid_indices]
    selected_mask = mask[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    selected_grid_points = grid_points[valid_indices][selected_mask]

    return selected_grid_points * np.array(spacing)

def sample_grid_volume(mask, spacing, resolution=15):
    mesh = volume_to_mesh(mask, spacing, threshold=0, method='surface_nets')
    bounds = mesh.bounds
    # Generate grid points within the bounding box
    x = np.arange(bounds[0], bounds[1], resolution)
    y = np.arange(bounds[2], bounds[3], resolution)
    z = np.arange(bounds[4], bounds[5], resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    grid_points_poly = pv.PolyData(grid_points)
    select = grid_points_poly.select_enclosed_points(mesh, check_surface=True, tolerance=1e-6)
    selected_grid_points = grid_points[select['SelectedPoints'].astype(bool)]

    return selected_grid_points

def volume_to_mesh(
    volume,
    spacing,
    method: str,  # Either `surface_nets` or `marching_cubes`
    threshold: float = 300,  # Min value for marching cubes (Hounsfield units)
    verbose: bool = True,  # Display progress bars for mesh processing steps
):
    """
    Convert the CT in a DRR object into a mesh.

    If using `method=="surface_nets"`, ensure you have `pyvista>=0.43` and `vtk>=9.3` installed.

    The mesh processing steps are:

    1. Keep only largest connected components
    2. Smooth
    3. Decimate (if `method=="marching_cubes"`)
    4. Fill any holes
    5. Clean (remove any redundant vertices/edges)
    """
    volume = np.flip(volume, axis=0)
    # Turn the CT into a PyVista object and run marching cubes
    grid = pv.ImageData(
        dimensions=volume.shape,
        spacing=spacing,
        origin=(0, 0, 0),
    )

    if method == "marching_cubes":
        mesh = grid.contour(
            isosurfaces=1,
            scalars=volume.flatten(order="F"),
            rng=[threshold, torch.inf],
            method="marching_cubes",
            progress_bar=verbose,
        )
    elif method == "surface_nets":
        grid.point_data["values"] = (
            volume.flatten(order="F") > threshold
        )
        try:
            mesh = grid.contour_labeled(smoothing=True, progress_bar=verbose)
        except AttributeError as e:
            raise AttributeError(
                f"{e}, ensure you are using pyvista>=0.43 and vtk>=9.3"
            )
    else:
        raise ValueError(
            f"method must be `marching_cubes` or `surface_nets`, not {method}"
        )

    # Process the mesh
    mesh.extract_largest(inplace=True, progress_bar=verbose)
    mesh.point_data.clear()
    mesh.cell_data.clear()
    mesh.smooth_taubin(
        n_iter=100,
        feature_angle=120.0,
        boundary_smoothing=False,
        feature_smoothing=False,
        non_manifold_smoothing=True,
        normalize_coordinates=True,
        inplace=True,
        progress_bar=verbose,
    )
    if method == "marching_cubes":
        mesh.decimate_pro(0.25, inplace=True, progress_bar=verbose)
    mesh.fill_holes(100, inplace=True, progress_bar=verbose)
    mesh.clean(inplace=True, progress_bar=verbose)
    return mesh