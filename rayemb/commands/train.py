import click
from datetime import datetime
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint 
import wandb

from rayemb.models import RayEmb, FixedLandmark
from rayemb.dataset import (
    SyntheticDeepFluoroDataModule,
    SyntheticCTPelvic1KDataModule,
)
from rayemb.utils import cosine_similarity, setup_logger

logger = setup_logger(__name__)
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@click.group()
def train():
    pass

@train.group()
def arbitrary_landmark():
    pass

# Define a function to hold common options
def common_options(func):
    options = [
        click.option('--data_dir', type=str, default='./data/ctpelvic1k_synthetic', help='Directory containing the dataset'),
        click.option('--template_dir', type=str, default='./data/ctpelvic1k_templates', help='Directory containing the templates'),
        click.option('--image_size', type=int, default=224, help='Image size'),
        click.option('--num_samples', type=int, default=30, help='Number of samples'),
        click.option('--sampling_distance', type=int, default=1, help='Sampling distance'),
        click.option('--batch_size', type=int, default=10, help='Batch size'),
        click.option('--num_workers', type=int, default=4, help='Number of workers for data loading'),
        click.option('--num_templates', type=int, default=4, help='Number of templates to use'),
        click.option('--lr', type=float, default=1e-4, help='Learning rate'),
        click.option('--emb_dim', type=int, default=32, help='Embedding dimension'),
        click.option('--max_epochs', type=int, default=100, help='Maximum number of epochs to train for'),
        click.option('--devices', type=int, default=1, help='Number of GPUs to use for training'),
        click.option('--precision', type=int, default=32, help='Precision for training'),
        click.option('--log_every_n_steps', type=int, default=10, help='Log metrics every n steps'),
        click.option('--val_check_interval', type=float, default=0.5, help='Run validation every fraction of the training data'),
        click.option('--gradient_clip_val', type=float, default=10.0, help='Gradient clipping value'),
        click.option('--temperature', type=float, default=1e-4, help='Temperature for similarity function'),
        click.option('--checkpoint_path', type=str, default='', help='Model checkpoint path'),
        click.option('--project', type=str, default='rayemb', help='Weights & Biases project name'),
    ]
    for option in options:
        func = option(func)
    return func

def train_rayemb(data_module, args):
    # Initialize Weights & Biases Logger
    checkpoint_dir = f'./checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    config = vars(args)
    config['checkpoint_dir'] = checkpoint_dir
    wandb_logger = WandbLogger(project=args.project, config=config)

    # Configure model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_localization_error',        # Metric to monitor for determining the best model
        dirpath=checkpoint_dir,  # Directory where model checkpoints will be saved
        filename='rayemb-{epoch:02d}-{val_localization_error:.2f}',  # Checkpoint file naming scheme
        save_top_k=3,              # Save the top 3 models based on the metric monitored
        mode='min',                # Mode 'min' indicates "less is better" for the monitored metric
        auto_insert_metric_name=False  # Avoid automatic insertion of the metric name in the filename
    )
    # Set up Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,       # Pass the WandB logger to the trainer
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,             # Maximum number of epochs to train for
        devices=args.devices,                 # Number of GPUs to use for training
        precision=args.precision,  # Use 16-bit precision if GPU available
        log_every_n_steps=args.log_every_n_steps,       # Log metrics every n steps
        val_check_interval=args.val_check_interval,    # Run validation every fraction of the training data
        gradient_clip_val=args.gradient_clip_val
    )
    num_negs = args.image_size**2 - 1
    if args.checkpoint_path != '' and os.path.exists(args.checkpoint_path):
        rayemb_model = RayEmb.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            map_location=torch.device("cuda:0"),
            similarity_fn=cosine_similarity,
            lr=args.lr, 
            num_negs=num_negs,
            temperature=args.temperature,
            image_size=args.image_size,
            emb_dim=args.emb_dim,
        )
    else:
        rayemb_model = RayEmb(
            similarity_fn=cosine_similarity,
            lr=args.lr, 
            num_negs=num_negs,
            image_size=args.image_size,
            emb_dim=args.emb_dim,
            temperature=args.temperature,
        )

    trainer.fit(rayemb_model, data_module)

@arbitrary_landmark.command()
@common_options
@click.option('--h5_dir', type=str, default='data/ipcai_2020_full_res_data.h5', help='Directory containing the h5 files')
def deepfluoro(**args):
    config = Config(**args)
    data_module = SyntheticDeepFluoroDataModule(
        data_dir=config.data_dir,           # Directory containing the dataset
        image_size=config.image_size,             # Image size
        template_dir=config.template_dir,
        h5_dir=config.h5_dir,           # Directory containing the h5 files
        num_samples=config.num_samples,
        num_templates=config.num_templates,
        sampling_distance=config.sampling_distance,
        batch_size=config.batch_size,              # Batch size
        num_workers=config.num_workers,             # Number of workers for data loading,
        sample_only_visible=True
    )
    train_rayemb(data_module, config)


@arbitrary_landmark.command()
@common_options
@click.option('--vol_dir', type=str, default='data/CTPelvic1K/dataset6_volume', help='Directory containing the volume files')
@click.option('--mask_dir', type=str, default='data/CTPelvic1K/dataset6_label', help='Directory containing the mask files')
def ctpelvic1k(**args):
    config = Config(**args)
    data_module = SyntheticCTPelvic1KDataModule(
        data_dir=config.data_dir,           # Directory containing the dataset
        image_size=config.image_size,             # Image size
        template_dir=config.template_dir,
        vol_dir=config.vol_dir,           # Directory containing the volume files
        mask_dir=config.mask_dir,           # Directory containing the mask files
        num_samples=config.num_samples,
        num_templates=config.num_templates,
        sampling_distance=config.sampling_distance,
        batch_size=config.batch_size,              # Batch size
        num_workers=config.num_workers,             # Number of workers for data loading,
        sample_only_visible=True
    )
    train_rayemb(data_module, config)
