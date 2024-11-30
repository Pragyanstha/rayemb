import torch

def l2_similarity(x: torch.Tensor, y: torch.Tensor, gamma=0.1) -> torch.Tensor:
    # Calculate the L2 norm squared
    dist_squared = torch.sum((x - y) ** 2, dim=-1)
    # Convert distance to a similarity measure
    similarity = -gamma * dist_squared  # or torch.exp(-dist_squared)
    return similarity

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)

def dot_product_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1)
