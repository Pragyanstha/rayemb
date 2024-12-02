import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
from torchvision.utils import make_grid
import torch.nn.functional as F
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rayemb.utils import find_batch_peak_coordinates
from rayemb.models.dinov2 import DinoV2
from rayemb.utils import infonce_loss

class RayEmb(pl.LightningModule):
    def __init__(self, 
                 similarity_fn, 
                 lr=1e-3, 
                 image_size=512, 
                 num_negs=1024, 
                 emb_dim=16, 
                 temperature=1e-1,
                 ):
        super(RayEmb, self).__init__()
        self.image_size = image_size
        self.lr = lr
        self.num_negs = num_negs
        self.log_img_interval = 1
        self.model = DinoV2(emb_dim=emb_dim)
        self.emb_dim = emb_dim
        self.training_step_outputs = None
        self.validation_step_outputs = None
        self.similarity_fn = similarity_fn
        self.temperature = temperature
        self.loss_fn = infonce_loss

    def forward_features(self, imgs, templates):
        """
        Compute the features of the image and templates.
        img: [B, 3, 224, 224]
        templates: [B, T, 3, 224, 224]
        Returns:
        img_descriptors: [B, 224, 224, 384]
        template_descriptors: [B*T, 16, 16, 384] 
        """
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

    def forward(self, batch):
        imgs = batch['img'] # [B, 3, 224, 224]
        B = imgs.shape[0]
        templates = batch['templates'] # [B, T, 3, 224, 224]
        T = templates.shape[1]
        proj_point_templates = batch['proj_point_templates'] # [B, N, T, 2]
        N = proj_point_templates.shape[1]
        proj_point_templates = rearrange(proj_point_templates, 'b n t c -> (b t) n c')
        proj_point_img_ = batch['proj_point_img'] # [B, N, 2]

        img_descriptors, template_descriptors = self.forward_features(imgs, templates) # [B, 224, 224, D], [B*T, 224, 224, D]
        # Get the template point descriptors
        # Convert the projected points to patches
        proj_point_templates.clip_(0, self.image_size - 1)
        # rescale to -1 to 1
        proj_point_templates = (proj_point_templates / (self.image_size - 1)) * 2 - 1 # [B*T, N , 2]

        # Grid sample template descriptors using proj_point_templates
        template_descriptors = template_descriptors.permute(0, 3, 1, 2) # [B*T, D, 224, 224]
        selected_template_descriptors = torch.nn.functional.grid_sample(
            template_descriptors,
            proj_point_templates[:, :, None, :], # [B*T, N, 1, 2]
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).permute(2, 0, 1) # [N, B*T, D]
        selected_template_descriptors = rearrange(selected_template_descriptors, 'n (b t) c -> b n t c', b=B, t=T) # [B, N, T, D]
        try:
            subpsace_projections = self.calc_subspace_matrix(selected_template_descriptors)
        except Exception as e:
            print(f"Error: {e}")
            return None

        anchor_projection_matrices = subpsace_projections # [B, N, D, D]
        query_features = rearrange(img_descriptors, 'b h w d -> b d (h w)')
        query_features = query_features.unsqueeze(1).repeat(1, N, 1, 1) # [B, N, D, (H W)]
        projected_features = anchor_projection_matrices @ query_features # [B, N, D, (H W)]

        query_features = rearrange(query_features, 'b n d e -> (b n) e d')
        projected_features = rearrange(projected_features, 'b n d e -> (b n) e d')
        # select positives using the proj_point_img
        # Find index of visible points
        proj_point_img_ = proj_point_img_.clip_(0, self.image_size - 1)
        proj_point_img = proj_point_img_.long()
        proj_point_img = rearrange(proj_point_img, 'b n c -> (b n) 1 c')
        # convert proj_point_img from pixel coordinates to index
        proj_point_img = proj_point_img[:, :, 1] * self.image_size + proj_point_img[:, :, 0] # [B*N, 1]

        sim = self.similarity_fn(query_features, projected_features)

        loss = self.loss_fn(sim, proj_point_img, temperature=self.temperature)

        similarity = sim.reshape(B, N, self.image_size, self.image_size)

        pred_proj_points = find_batch_peak_coordinates(rearrange(similarity, 'b n h w -> (b n) h w')) # [B*N, 2]
        pred_proj_points = rearrange(pred_proj_points, '(b n) c -> b n c', b=B, n=N)

        localization_error = torch.norm(proj_point_img_ - pred_proj_points, p=2, dim=-1).mean()

        return {
            'loss': loss,
            'pred_heatmaps': similarity,
            'proj_point_img': proj_point_img_,
            'localization_error': localization_error
        }

    def inference(self, 
                  img, 
                  templates, 
                  sample_points, 
                  proj_matrices, 
                  template_sensor_size=512, 
                  return_features=False):
        """
        img: [3, 224, 224]
        templates: [T, 3, 224, 224]
        sample_points: [N, 3]
        proj_matrices: [T, 3, 4]
        """
        T = templates.shape[0]
        N = sample_points.shape[0]
        img = img.unsqueeze(0)
        templates = templates.unsqueeze(0)
        with torch.no_grad():
            img_descriptors, template_descriptors = self.forward_features(img, templates) # [B, 224, 224, D], [B*T, 224, 224, D]

            # Calculate proj_point_templates
            proj_point_templates = torch.matmul(proj_matrices, torch.cat([sample_points, torch.ones(sample_points.shape[0], 1).to(sample_points.device)], dim=1).T[None, ...])
            proj_point_templates = proj_point_templates[:,:,:].permute(0, 2, 1) # (T, N, 3)
            proj_point_templates = proj_point_templates[:, :, :2] / proj_point_templates[:, :, 2:]
            proj_point_templates = proj_point_templates / template_sensor_size * self.image_size

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

    def log_heatmaps(self, pred_heatmaps, gt_proj_points, phase="train"):
        """
        Log heatmaps to Weights & Biases with ground truth projection points plotted.
        
        Args:
        pred_heatmaps (torch.Tensor): Predicted heatmaps tensor [B, H, W].
        gt_proj_points (torch.Tensor): Ground truth projection points tensor [B, 2] (assumed [x, y] format).
        step (int): Current step or batch_idx.
        phase (str): Phase of training, e.g., 'train', 'val'.
        """
        # Convert PyTorch tensors to numpy arrays for visualization
        if torch.is_tensor(pred_heatmaps):
            # pred_heatmaps = torch.sigmoid(pred_heatmaps)  # Convert logits to probabilities if not done earlier
            pred_points_ = find_batch_peak_coordinates(pred_heatmaps).cpu().detach().numpy()  # Find peak coordinates in heatmaps
            pred_heatmaps = pred_heatmaps.cpu().detach().numpy()

        if torch.is_tensor(gt_proj_points):
            gt_proj_points = gt_proj_points.cpu().detach().numpy()

        # Create figure for plotting
        figs = []
        for i in range(pred_heatmaps.shape[0]):
            fig, ax = plt.subplots()
            heatmap = pred_heatmaps[i]
            gt_points = gt_proj_points[i]
            pred_points = pred_points_[i]    

            # Normalize heatmap for colormap application
            heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            colored_heatmap = plt.cm.jet(heatmap_norm)  # Apply jet colormap

            # Plot heatmap
            ax.imshow(colored_heatmap, interpolation='nearest', origin='upper')

            # Plot points - assume points are in [x, y] format and need scaling according to heatmap size
            ax.scatter(gt_points[0], gt_points[1], color='cyan', s=40, edgecolors='w')  # adjust size and color as needed
            ax.scatter(pred_points[0], pred_points[1], color='magenta', s=40, edgecolors='w')  # adjust size and color as needed

            # Remove axes and ticks
            ax.axis('off')

            # Save fig to a numpy array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            figs.append(data)
            plt.close(fig)

        # Convert list of numpy arrays to torch tensor and make a grid
        tensor_figs = torch.from_numpy(np.array(figs)).permute(0, 3, 1, 2).float()  # Convert to C, H, W format and ensure it's float

        # No need to convert to int before making grid
        grid = make_grid(tensor_figs, nrow=10, normalize=True, scale_each=True)

        # Log to WandB
        self.logger.experiment.log({
            f"{phase}_predictions": wandb.Image(grid),
        })
        # delete the tensor_figs
        del tensor_figs
        del grid

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_localization_error', outputs['localization_error'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs = outputs
        return loss
    
    def on_train_epoch_end(self):
        # Log heatmaps at the end of the epoch
        outputs = self.training_step_outputs
        self.log_heatmaps(outputs['pred_heatmaps'][0], outputs['proj_point_img'][0], "train")

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        # loss_error = outputs['loss_error']
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_localization_error', outputs['localization_error'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs = outputs
        return loss
    
    def on_validation_epoch_end(self):
        # Log heatmaps at the end of the epoch
        outputs = self.validation_step_outputs
        self.log_heatmaps(outputs['pred_heatmaps'][0], outputs['proj_point_img'][0], "val")

    def configure_optimizers(self):
        # Filter parameters with requires_grad enabled
        optimizer_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(optimizer_parameters, lr=self.lr)

        # Set up the Cosine Annealing scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        # In PyTorch Lightning, return a dictionary when using both optimizer and scheduler
        return {
            'optimizer': optimizer,
        }
