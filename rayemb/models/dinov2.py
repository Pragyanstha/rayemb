import torch
import torch.nn as nn
from einops import rearrange


class DinoV2(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", 'dinov2_vits14', pretrained=True)
        self.linear = nn.Linear(384, emb_dim)

    def forward(self, x):
        res = self.dinov2_model.forward_features(x)
        patch_embeddings = res["x_norm_patchtokens"]
        ray_embeddings = self.linear(patch_embeddings)
        ray_embeddings = rearrange(ray_embeddings, 'b (h w) c -> b c h w', h=16, w=16)
        return ray_embeddings
