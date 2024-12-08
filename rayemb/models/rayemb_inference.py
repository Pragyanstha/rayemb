import pytorch_lightning as pl
import torch
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms as T
import json
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rayemb.utils import find_batch_peak_coordinates
from rayemb.models.dinov2 import DinoV2

class RayEmbSubspaceInference(pl.LightningModule):
    def __init__(self, 
                 similarity_fn, 
                 image_size=512, 
                 num_templates=4,
                 emb_dim=16, 
                 temperature=1e-1,
                 template_root_dir="data/ctpelvic1k_templates_v2"
                 ):
        super(RayEmbSubspaceInference, self).__init__()
        self.image_size = image_size
        self.model = DinoV2(emb_dim=emb_dim)
        self.similarity_fn = similarity_fn
        self.num_templates = num_templates
        self.temperature = temperature
        self.no_aug_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ])
        self.img = None
        self.templates = None
        self.proj_matrices = None
        self.template_root_dir = template_root_dir

    def forward_features(self, imgs, templates):
        """
        Compute the features of the image and templates.
        img: [B, 3, 224, 224]
        templates: [B, T, 3, 224, 224]
        Returns:
        img_descriptors: [B, 224, 224, 384]
        template_descriptors: [B*T, 16, 16, 384] 
        """
        print("imgs.shape", imgs.shape)
        print("templates.shape", templates.shape)
        templates = rearrange(templates, 'b t c h w -> (b t) c h w')

        img_features = self.model(imgs)
        template_features = self.model(templates)

        img_features = F.interpolate(img_features, size=(self.image_size, self.image_size), mode='bilinear')

        img_features = rearrange(img_features, 'b c h w -> b h w c')
        template_features = rearrange(template_features, 'n c h w -> n h w c')
        return img_features, template_features

    def calc_subspace_matrix(self, selected_template_descriptors):
        """
        selected_template_descriptors: [B, N, T, D]
        """
        selected_template_descriptors = rearrange(selected_template_descriptors, 'b n t d -> b n d t')
        pinv = torch.linalg.pinv(selected_template_descriptors)
        P = selected_template_descriptors @ pinv
        return P

    def set_image(self, img):
        img = self.no_aug_transforms(img[..., :3])
        img = img.unsqueeze(0).float()
        self.img = img.to(self.device)

    def set_templates(self, template_path):
        """
        Load the templates and projection matrices for the given specimen_id
        """
        temp_dir = f"{template_path}/info.json"
        with open(temp_dir, 'r') as f:
            loaded_template = json.load(f)
            self.template_sensor_size_width = loaded_template['params']['sensor_width']
            self.template_sensor_size_height = loaded_template['params']['sensor_height']
            loaded_template.pop('params')

        templates = []
        proj_matrices = []
        keys = list(loaded_template.keys())
        keys = np.random.choice(keys, self.num_templates, replace=False)
        for k in keys:
            temp = imageio.imread(f'{template_path}/images/{k}.png')
            temp = self.no_aug_transforms(temp)
            temp = temp.unsqueeze(0)
            proj_matrix = np.array(loaded_template[k]['proj'])
            proj_matrix = torch.tensor(proj_matrix)
            proj_matrix = proj_matrix.unsqueeze(0)
            proj_matrices.append(proj_matrix)
            templates.append(temp)
        self.templates = torch.cat(templates, dim=0).float().unsqueeze(0).to(self.device) # (T, C, H, W)
        self.proj_matrices = torch.cat(proj_matrices, dim=0).float().unsqueeze(0).to(self.device) # (T, 3, 4)
        print(f"Loaded {len(templates)} templates and {len(proj_matrices)} projection matrices")

    def calc_features(self):
        with torch.no_grad():
            img_descriptors, template_descriptors = self.forward_features(self.img, self.templates) # [B, 224, 224, D], [B*T, 224, 224, D]
            self.img_descriptors = img_descriptors
            self.template_descriptors = template_descriptors

    def inference(self, sample_points, return_features=False):
        """
        img: [3, 224, 224]
        templates: [T, 3, 224, 224]
        sample_points: [N, 3]
        proj_matrices: [T, 3, 4]
        """
        if self.templates is None:
            raise ValueError("Templates not set")
        if self.proj_matrices is None:
            raise ValueError("Projection matrices not set")
        if self.img_descriptors is None:
            raise ValueError("Image descriptors not set")
        if self.template_descriptors is None:
            raise ValueError("Template descriptors not set")
        if self.img is None:
            raise ValueError("Image not set")
        templates = self.templates.squeeze(0)
        proj_matrices = self.proj_matrices.squeeze(0)

        T = templates.shape[0]
        N = sample_points.shape[0]

        img_descriptors = self.img_descriptors
        template_descriptors = self.template_descriptors

        # Calculate proj_point_templates
        proj_point_templates = torch.matmul(proj_matrices, torch.cat([sample_points, torch.ones(sample_points.shape[0], 1).to(sample_points.device)], dim=1).T[None, ...])
        proj_point_templates = proj_point_templates[:,:,:].permute(0, 2, 1) # (T, N, 3)
        proj_point_templates = proj_point_templates[:, :, :2] / proj_point_templates[:, :, 2:]
        proj_point_templates = proj_point_templates / self.template_sensor_size_width * self.image_size

        proj_point_templates.clip_(0, self.image_size-1)
        # rescale to -1 to 1
        proj_point_templates = (proj_point_templates / (self.image_size-1)) * 2 - 1 # [T, N , 2]

        # Grid sample template descriptors using proj_point_templates
        template_descriptors = template_descriptors.permute(0, 3, 1, 2) # [B*T, D, 224, 224]
        selected_template_descriptors = torch.nn.functional.grid_sample(
            template_descriptors,
            proj_point_templates[:, :, None, :], # [B*T, N, 1, 2]
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).permute(2, 0, 1) # [N, B*T, D]
        selected_template_descriptors = rearrange(selected_template_descriptors, 'n (b t) c -> b n t c', b=1, t=T) # [B, N, T, D]
        subpsace_projections = self.calc_subspace_matrix(selected_template_descriptors)

        anchor_projection_matrices = subpsace_projections # [B, N, D, D]
        query_features = rearrange(img_descriptors, 'b h w d -> b d (h w)')
        query_features = query_features.unsqueeze(1).repeat(1, N, 1, 1) # [B, N, D, (H W)]
        projected_features = anchor_projection_matrices @ query_features # [B, N, D, (H W)]

        query_features = rearrange(query_features, 'b n d e -> (b n) e d')
        projected_features = rearrange(projected_features, 'b n d e -> (b n) e d')

        sim = self.similarity_fn(query_features, projected_features)
        
        if return_features:
            return sim.reshape(N, self.image_size, self.image_size), query_features, projected_features, subpsace_projections[0]
        else:
            return sim.reshape(N, self.image_size, self.image_size)
