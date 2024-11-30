import numpy as np
import torch


def generate_gaussian_heatmap(xy, height, width, sigma_x=10.0, sigma_y=10.0):
    """
    Generate a Gaussian heatmap using the Gaussian function.

    Args:
    xy (tuple): A tuple (x, y) representing the center coordinates on the image.
    height (int): The height of the output heatmap.
    width (int): The width of the output heatmap.
    sigma_x (float): The standard deviation of the Gaussian along the x-axis.
    sigma_y (float): The standard deviation of the Gaussian along the y-axis.

    Returns:
    numpy.ndarray: A height x width heatmap with a Gaussian peak at (x, y).
    """
    # Create a grid of (x, y) coordinates
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(xs, ys)

    # Center coordinates
    x, y = xy

    # Calculate the Gaussian heatmap
    heatmap = np.exp(-(((X - x) ** 2) / (2 * sigma_x ** 2) + ((Y - y) ** 2) / (2 * sigma_y ** 2)))

    # Mask out values outside of 2 sigma range
    # in circular mask
    mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) < 3 * max(sigma_x, sigma_y)
    heatmap *= mask

    # Normalize to 0 - 1
    # if np.max(heatmap) > 0:  # Check to avoid division by zero if all values are masked out
    #     heatmap = heatmap / np.max(heatmap)
    return heatmap

def find_batch_peak_coordinates(heatmaps):
    """
    Find the coordinates of the peak (maximum value) in a batch of Gaussian heatmaps.

    Args:
    heatmaps (torch.Tensor): A 3D tensor where the first dimension is the batch size,
                             and the next two dimensions represent the Gaussian heatmap.

    Returns:
    torch.Tensor: A tensor containing the (x, y) coordinates of the peak for each heatmap in the batch.
    """
    # Find the index of the maximum value in each heatmap in the batch
    batch_indices = torch.argmax(heatmaps.view(heatmaps.size(0), -1), dim=1)
    
    # Convert flat indices to 2D indices
    y_indices = batch_indices // heatmaps.size(2)
    x_indices = batch_indices % heatmaps.size(2)
    
    # Stack coordinates in the order (x, y) for each heatmap
    peak_coordinates = torch.stack((x_indices, y_indices), dim=1)
    
    return peak_coordinates

def find_batch_peak_coordinates_maxval(heatmaps):
    """
    Find the coordinates of the peak (maximum value) in a batch of Gaussian heatmaps.

    Args:
    heatmaps (torch.Tensor): A 3D tensor where the first dimension is the batch size,
                             and the next two dimensions represent the Gaussian heatmap.

    Returns:
    torch.Tensor: A tensor containing the (x, y) coordinates of the peak for each heatmap in the batch.
    """
    # Find the index of the maximum value in each heatmap in the batch
    batch_indices = torch.argmax(heatmaps.view(heatmaps.size(0), -1), dim=1)
    
    # Convert flat indices to 2D indices
    y_indices = batch_indices // heatmaps.size(2)
    x_indices = batch_indices % heatmaps.size(2)
    
    # Stack coordinates in the order (x, y) for each heatmap
    peak_coordinates = torch.stack((x_indices, y_indices), dim=1)

    max_values = torch.max(heatmaps.view(heatmaps.size(0), -1), dim=1)[0]

    return peak_coordinates, max_values

