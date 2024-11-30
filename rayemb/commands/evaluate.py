import click
import os
import json
import time

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc
import nibabel as nib

from rayemb.models import RayEmb, FixedLandmark
from rayemb.dataset import SyntheticCTPelvic1KDataset, RealDeepFluoroDataset
from rayemb.registration import PnPRunner, RayEmbCorrespondenceSampler
from rayemb.utils import cosine_similarity, NumpyEncoder, setup_logger
from rayemb.constants import TOP_K
from rayemb.utils.evaluator import Evaluator


matplotlib.use('TkAgg')
logger = setup_logger(__name__)

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@click.group()
def evaluate():
    pass

@evaluate.group()
def arbitrary_landmark():
    pass

# Define a function to hold common options
def common_options(func):
    options = [
        click.option('--data_dir', type=str, default='./data/ctpelvic1k_synthetic', help='Directory containing the dataset'),
        click.option('--template_dir', type=str, default='./data/deepfluoro_templates', help='Directory containing the templates'),
        click.option('--image_size', type=int, default=224, help='Image size'),
        click.option('--num_templates', type=int, default=4, help='Number of templates to use'),
        click.option('--emb_dim', type=int, default=32, help='Embedding dimension'),
        click.option('--device', type=str, default='cuda:0', help='Device to use for training'),
        click.option('--num_samples', type=int, default=10000, help='Number of samples'),
        click.option('--sampling_distance', type=int, default=1, help='Sampling distance'),
        click.option('--checkpoint_path', type=str, default='./checkpoints/rayemb-ctpelvic1k.ckpt', help='Model checkpoint path'),
        click.option('--output_dir', type=str, default='./results', help='Output directory'),
    ]
    for option in options:
        func = option(func)
    return func

def evaluate_rayemb(dataset, args):
    basefolderdir = os.path.join(args.output_dir, f"{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(basefolderdir, exist_ok=True)
    model = RayEmb.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        map_location=torch.device(args.device),
        similarity_fn=cosine_similarity,
        image_size=args.image_size,
        emb_dim=args.emb_dim,
    )
    model.eval()
    correspondence_sampler = RayEmbCorrespondenceSampler(model, top_k=TOP_K)
    pnp_runner = PnPRunner(correspondence_sampler=correspondence_sampler, 
                           device=args.device, 
                           is_real=dataset.__class__.__name__ == "RealDeepFluoroDataset")
    evaluator = Evaluator(model, pnp_runner, args.device)
    for data in dataset:
        logger.info(f"Evaluating {data['specimen_id']} - {data['img_id']}")
        folderdir = os.path.join(basefolderdir, f"{data['specimen_id']}", f"{data['img_id']}")
        os.makedirs(folderdir, exist_ok=True)
        if dataset.__class__.__name__ == "SyntheticCTPelvic1KDataset":
            specimen = None
            volume = nib.load(f"{args.vol_dir}/{data['specimen_id']}.nii.gz").get_fdata()
            voxel_spacings = np.array(nib.load(f"{args.vol_dir}/{data['specimen_id']}.nii.gz").header.get_zooms())
        else:
            volume = None
            voxel_spacings = None
            specimen = dataset.specimen

        json_data, p, fig  = evaluator(data, volume=volume, voxel_spacings=voxel_spacings, specimen=specimen, visualize=True, off_screen=True)
        
        with open(os.path.join(folderdir, "camera.json"), "w") as f:
            json.dump(json_data, f, indent=4, cls=NumpyEncoder)
        fig.savefig(os.path.join(folderdir, "projections.png"))
        plt.close(fig)
        p.save_graphic(os.path.join(folderdir, "registration.svg"))
        p.close()

@arbitrary_landmark.command()
@common_options
@click.option('--vol_dir', type=str, default='data/CTPelvic1K/dataset6_volume', help='Directory containing the volume files')
@click.option('--mask_dir', type=str, default='data/CTPelvic1K/dataset6_label', help='Directory containing the mask files')
def ctpelvic1k(**args):
    config = Config(**args)
    dataset = SyntheticCTPelvic1KDataset(
        data_dir=config.data_dir,           # Directory containing the dataset
        image_size=config.image_size,             # Image size
        template_dir=config.template_dir,
        vol_dir=config.vol_dir,           # Directory containing the volume files
        mask_dir=config.mask_dir,           # Directory containing the mask files
        num_samples=config.num_samples,
        num_templates=config.num_templates,
        sampling_distance=config.sampling_distance,
        sample_only_visible=False,
        split='test'
    )
    evaluate_rayemb(dataset, config)

@arbitrary_landmark.command()
@common_options
@click.option('--specimen_id', type=click.Choice(['specimen_1', 'specimen_2', 'specimen_3', 'specimen_4', 'specimen_5', 'specimen_6']), default="specimen_2", help='Specimen to use')
@click.option('--h5_dir', type=str, default='data/ipcai_2020_full_res_data.h5', help='Directory containing the h5 files')
def deepfluoro(**args):
    config = Config(**args)
    dataset = RealDeepFluoroDataset(
        image_size=config.image_size,             # Image size
        template_dir=config.template_dir,
        h5_dir=config.h5_dir,           # Directory containing the h5 files
        num_samples=config.num_samples,
        num_templates=config.num_templates,
        sampling_distance=config.sampling_distance,
        sample_only_visible=False,
        specimen_id=config.specimen_id
    )
    evaluate_rayemb(dataset, config)
