import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
from torchvision.utils import make_grid
from einops import rearrange
import kornia

from rayemb.models.unet import UNet
from rayemb.utils import generate_gaussian_heatmap, find_batch_peak_coordinates


class FixedLandmark(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(FixedLandmark, self).__init__()
        self.input_dim = 3
        self.unet = UNet(n_channels=self.input_dim, n_classes=14)

        self.loss_bce = torch.nn.BCEWithLogitsLoss()
        self.loss_mse = torch.nn.MSELoss()
        self.lr = lr
        self.log_img_interval = 1
        self.training_step_outputs = None
        self.validation_step_outputs = None

    def forward(self, batch):
        imgs = batch['img'] # [B, 3, 224, 224]
        B = imgs.shape[0]
        gt_proj_landmarks = batch['gt_proj_landmarks'] # [B, N, 2]
        N = gt_proj_landmarks.shape[1]

        preds = self.unet(imgs)# [B, N, 224, 224]
        pred_heatmaps_batched = rearrange(preds, 'b n h w -> (b n) h w')

        # Calculate errors for the proj_point_img
        pred_proj_point_img = find_batch_peak_coordinates(pred_heatmaps_batched).float()
        pred_proj_point_img = rearrange(pred_proj_point_img, '(b n) c -> b n c', b=B, n=N)

        # Get gaussian heatmaps for proj_point_img
        gt_heatmaps = []
        for i in range(B):
            gt_heatmaps_batch = []
            for j in range(N):
                gt_heatmap = generate_gaussian_heatmap(
                    (gt_proj_landmarks[i, j, 0].item(), gt_proj_landmarks[i, j, 1].item()), 224, 224, sigma_x=1.0, sigma_y=1.0)
                gt_heatmaps_batch.append(torch.from_numpy(gt_heatmap))
            gt_heatmaps.append(torch.stack(gt_heatmaps_batch, dim=0))
        gt_heatmaps = torch.stack(gt_heatmaps, dim=0).float().to(preds.device)

        loss = self.loss_bce(preds, gt_heatmaps)

        return {
            "pred_heatmaps": preds,
            "gt_heatmaps": gt_heatmaps,
            "loss": loss,
            "pred_proj_point_img": pred_proj_point_img,
        }

    def inference(self, img):
        pred_heatmaps = self.unet(img.unsqueeze(0)).squeeze(0) # [1, N, 224, 224]
        pred_heatmaps = torch.sigmoid(pred_heatmaps)

        # Calculate errors for the proj_point_img
        pred_proj_point_img = find_batch_peak_coordinates(pred_heatmaps).float()

        return {
            "pred_heatmaps": pred_heatmaps,
            "pred_proj_points": pred_proj_point_img,
        }

    def log_heatmaps(self, pred_heatmaps, gt_heatmaps,
                    step, phase):
        """
        Log heatmaps to Weights & Biases.
        
        Args:
        pred_heatmaps (torch.Tensor): Predicted heatmaps tensor.
        gt_heatmaps (torch.Tensor): Ground truth heatmaps tensor.
        step (int): Current step or batch_idx.
        """
        # Convert PyTorch tensors to numpy arrays for visualization
        if torch.is_tensor(pred_heatmaps):
            # Apply sigmoid to convert logits to probabilities
            # pred_heatmaps = torch.sigmoid(pred_heatmaps)
            pred_heatmaps = kornia.geometry.spatial_softmax2d(pred_heatmaps.unsqueeze(0))[0]
            pred_heatmaps = pred_heatmaps.cpu().detach().numpy()
        if torch.is_tensor(gt_heatmaps):
            gt_heatmaps = gt_heatmaps.cpu().detach().numpy()

        # Assuming single-channel heatmaps, add color channel for compatibility
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=1)
        gt_heatmaps = np.expand_dims(gt_heatmaps, axis=1)

        # Use torchvision's make_grid to create a grid of images
        pred_grid = make_grid(torch.from_numpy(pred_heatmaps), nrow=10, normalize=True, scale_each=True)
        gt_grid = make_grid(torch.from_numpy(gt_heatmaps), nrow=10, normalize=True, scale_each=True)

        # Log heatmaps to WandB
        self.logger.experiment.log({
            f"{phase}_predictions": wandb.Image(pred_grid),
            f"{phase}_ground_truth": wandb.Image(gt_grid)
        })

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs = outputs
        return loss
    
    def on_train_epoch_end(self):
        # Log heatmaps at the end of the epoch
        outputs = self.training_step_outputs
        self.log_heatmaps(outputs['pred_heatmaps'][0], outputs['gt_heatmaps'][0], self.global_step, "train")

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        # loss_error = outputs['loss_error']
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_loss_error', loss_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs = outputs
        return loss
    
    def on_validation_epoch_end(self):
        # Log heatmaps at the end of the epoch
        outputs = self.validation_step_outputs
        self.log_heatmaps(outputs['pred_heatmaps'][0], outputs['gt_heatmaps'][0], self.global_step, "val")

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
