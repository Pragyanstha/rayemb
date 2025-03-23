import click
import os
import json
import torch
import numpy as np
import nibabel as nib
import cv2
import imageio
from tqdm import tqdm
import csv
import glob

from diffdrr.drr import DRR
from diffdrr.pose import convert

from rayemb.dataset import DeepFluoroDataset
from rayemb.utils import load, get_random_offset_params, get_random_offset, Transforms, sample_grid_from_mask, setup_logger
from rayemb.constants import ( 
    DEEPFLUORO_IMAGE_SIZE, 
    DEEPFLUORO_PIXEL_SIZE, 
    DEEPFLUORO_SOURCE_TO_DETECTOR_DISTANCE, 
    DEEPFLUORO_CROP_WINDOW 
)

logger = setup_logger(__name__)

@click.group()
def generate():
    pass

@generate.group()
def dataset():
    pass

@generate.group()
def template():
    pass


@dataset.command()
@click.option('--input_file', default="data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz", help='Path to the volume')
@click.option('--mask_dir', default=None, help='Path to the mask')
@click.option('--threshold', default=300, help='Threshold for the mask')
@click.option('--height', default=512, help='Image height')
@click.option('--num_samples', default=100, help='Number of samples')
@click.option('--source_to_detector_distance', default=1000, help='Source to detector distance')
@click.option('--pixel_size', default=0.5, help='Pixel size')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--output_dir', default='./data/ctpelvic1k_synthetic', help='Directory to save generated DRRs')
def custom(input_file, mask_dir, threshold, height, num_samples, device, output_dir, source_to_detector_distance, pixel_size):
    logger.info(f"Generating {num_samples} DRRs for file {input_file}...")
    filename = os.path.basename(input_file).split(".")[0]
    output_dir = os.path.join(output_dir, f"{filename}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cameras"), exist_ok=True)

    device = torch.device(device)

    logger.info(f"Loading volume for {filename}")
    filepath_sampled_points = os.path.join(output_dir, "sampled_points.npy")
    filepath_vol_landmarks = os.path.join(output_dir, "vol_landmarks.npy")
    nii_image = nib.load(input_file)
    volume = nii_image.get_fdata()
    voxel_spacings = np.array(nii_image.header.get_zooms())

    if mask_dir is not None:
        logger.info(f"Loading mask for {filename}")
        nifti_mask = nib.load(mask_dir)
        mask = (nifti_mask.get_fdata() > threshold).astype(np.float32)
    else:
        logger.info(f"Generating mask for {filename}")
        mask = (volume > threshold).astype(np.float32)
        logger.info(f"Found {np.sum(mask)} pixels in the mask")

    sampled_points = sample_grid_from_mask(mask, voxel_spacings, 1).astype(np.float32)
    # pick 100 points randomly from sampled_points
    vol_landmarks = sampled_points[np.random.choice(sampled_points.shape[0], 100, replace=False)]
    # save as npy
    np.save(filepath_sampled_points, sampled_points)
    np.save(filepath_vol_landmarks, vol_landmarks)
    logger.info(f"Finished Loading volume for {filename}")

    isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
    isocenter_xyz = torch.tensor(volume.shape) * voxel_spacings / 2
    isocenter_xyz = isocenter_xyz.unsqueeze(0)
    isocenter_pose = convert(
        isocenter_rot,
        isocenter_xyz,
        parameterization="euler_angles",
        convention="ZYX",
    ).to(device)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    # Use the same parameters for DRR rendering as DeepFluoro
    delx = pixel_size
    drr = DRR(
        volume,
        voxel_spacings,
        source_to_detector_distance / 2.0,
        height,
        delx,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        patch_size=128
    ).to(device)
    transforms = Transforms(height)
    
    metadata = {
        "params": {
            "vol_path": input_file,
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }

    # TODO: Make these parameters configurable
    params = {
        "r1": 0.3,
        "r2": 0.3,
        "r3": 0.3,
        "t1": (0, 5),
        "t2": (0, 5),
        "t3": (0, 5)
    }

    for counter in tqdm(range(num_samples)):
        contrast = contrast_distribution.sample().item()
        offset = get_random_offset_params(1, params, device)
        pose = isocenter_pose.compose(offset)
        img = drr(pose, bone_attenuation_multiplier=contrast)
        img = transforms(img)
        img = img.squeeze().cpu().numpy()
        
        output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
        img = cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imageio.imwrite(output_path, img)
        metadata["proj"] = [drr.compute_projection_matrix(pose).cpu().numpy().tolist()]
        metadata["pose"] = pose.matrix.cpu().numpy().tolist()

        metadata_path = os.path.join(output_dir, 'cameras', f'{counter:04d}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

@dataset.command()
@click.option('--id_number', default=2, help='Specimen number (1-6)')
@click.option('--height', default=512, help='Image height')
@click.option('--num_samples', default=100, help='Number of samples')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--input_file', default="./data/ipcai_2020_full_res_data.h5", help='Path to the volume')
@click.option('--output_dir', default='./data/deepfluoro_synthetic', help='Directory to save generated DRRs')
def deepfluoro(id_number, height, num_samples, device, output_dir, input_file):
    logger.info(f"Generating {num_samples} DRRs for specimen {id_number}...")
    output_dir = os.path.join(output_dir, f"specimen_{id_number}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cameras"), exist_ok=True)

    device = torch.device(device)
    specimen, _, transforms, drr = load(id_number, height, device)

    specimen = DeepFluoroDataset(id_number, filename=input_file, preprocess=True)
    isocenter_pose = specimen.isocenter_pose.to(device)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    subsample = (DEEPFLUORO_IMAGE_SIZE - DEEPFLUORO_CROP_WINDOW) / height
    delx = DEEPFLUORO_PIXEL_SIZE * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        DEEPFLUORO_SOURCE_TO_DETECTOR_DISTANCE,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        renderer="siddon",
    ).to(device)
    transforms = Transforms(height)
    
    metadata = {
        "params": {
            "vol_path": input_file,  # Update this path as needed
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }

    for counter in tqdm(range(num_samples)):
        contrast = contrast_distribution.sample().item()
        offset = get_random_offset(1, device)
        pose = isocenter_pose.compose(offset)
        img = drr(pose, bone_attenuation_multiplier=contrast)
        img = transforms(img)
        img = img.squeeze().cpu().numpy()
        
        output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
        img = cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imageio.imwrite(output_path, img)
        metadata["proj"] = [drr.compute_projection_matrix(pose).cpu().numpy().tolist()]
        metadata["pose"] = [pose.matrix.cpu().numpy().tolist()]

        metadata_path = os.path.join(output_dir, 'cameras', f'{counter:04d}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

@dataset.command()
@click.option('--input_file', default="data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz", help='Path to the volume')
@click.option('--mask_dir', default="data/CTPelvic1K/dataset6_label", help='Path to the mask')
@click.option('--height', default=512, help='Image height')
@click.option('--num_samples', default=100, help='Number of samples')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--output_dir', default='./data/ctpelvic1k_synthetic', help='Directory to save generated DRRs')
def ctpelvic1k(input_file, mask_dir, height, num_samples, device, output_dir ):
    logger.info(f"Generating {num_samples} DRRs for file {input_file}...")
    filename = os.path.basename(input_file).split(".")[0]
    output_dir = os.path.join(output_dir, f"{filename}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cameras"), exist_ok=True)

    device = torch.device(device)

    logger.info(f"Loading volume for {filename}")
    filepath_sampled_points = os.path.join(output_dir, "sampled_points.npy")
    filepath_vol_landmarks = os.path.join(output_dir, "vol_landmarks.npy")
    nii_image = nib.load(input_file)
    volume = nii_image.get_fdata()
    voxel_spacings = np.array(nii_image.header.get_zooms())

    maskname = "_".join(filename.split("_")[:3] + ["mask", "4label"]) + ".nii.gz"
    nifti_mask = nib.load(f"{mask_dir}/{maskname}")
    mask = (nifti_mask.get_fdata() > 0).astype(np.float32)
    sampled_points = sample_grid_from_mask(mask, voxel_spacings, 1).astype(np.float32)
    # pick 100 points randomly from sampled_points
    vol_landmarks = sampled_points[np.random.choice(sampled_points.shape[0], 100, replace=False)]
    # save as npy
    np.save(filepath_sampled_points, sampled_points)
    np.save(filepath_vol_landmarks, vol_landmarks)
    logger.info(f"Finished Loading volume for {filename}")

    isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
    isocenter_xyz = torch.tensor(volume.shape) * voxel_spacings / 2
    isocenter_xyz = isocenter_xyz.unsqueeze(0)
    isocenter_pose = convert(
        isocenter_rot,
        isocenter_xyz,
        parameterization="euler_angles",
        convention="ZYX",
    ).to(device)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    # Use the same parameters for DRR rendering as DeepFluoro
    subsample = (DEEPFLUORO_IMAGE_SIZE - DEEPFLUORO_CROP_WINDOW) / height
    delx = DEEPFLUORO_PIXEL_SIZE * subsample
    drr = DRR(
        volume,
        voxel_spacings,
        DEEPFLUORO_SOURCE_TO_DETECTOR_DISTANCE,
        height,
        delx,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        patch_size=128
    ).to(device)
    transforms = Transforms(height)
    
    metadata = {
        "params": {
            "vol_path": input_file,
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }

    for counter in tqdm(range(num_samples)):
        contrast = contrast_distribution.sample().item()
        offset = get_random_offset(1, device)
        pose = isocenter_pose.compose(offset)
        img = drr(pose, bone_attenuation_multiplier=contrast)
        img = transforms(img)
        img = img.squeeze().cpu().numpy()
        
        output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
        img = cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imageio.imwrite(output_path, img)
        metadata["proj"] = [drr.compute_projection_matrix(pose).cpu().numpy().tolist()]
        metadata["pose"] = pose.matrix.cpu().numpy().tolist()

        metadata_path = os.path.join(output_dir, 'cameras', f'{counter:04d}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


@template.command()
@click.option('--input_file', default="data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz", help='Path to the volume')
@click.option('--height', default=512, help='Image height')
@click.option('--alpha_min', default=np.pi/4.0, type=float, help='Minimum alpha angle')
@click.option('--alpha_max', default=np.pi*3.0/4.0, type=float, help='Maximum alpha angle')
@click.option('--beta_min', default=-np.pi/4.0, type=float, help='Minimum beta angle')
@click.option('--beta_max', default=np.pi/4.0, type=float, help='Maximum beta angle')
@click.option('--steps', default=4, help='Number of steps in alpha and beta ranges')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--source_to_detector_distance', default=1800, help='Source to detector distance')
@click.option('--pixel_size', default=1.4, help='Pixel size')
@click.option('--output_dir', default='./data/ctpelvic1k_templates', help='Directory to save generated DRRs')
def custom(input_file, height, alpha_min, alpha_max, beta_min, beta_max, steps, device, output_dir, pixel_size, source_to_detector_distance):
    logger.info(f"Generating DRRs for file {input_file}...")
    filename = os.path.basename(input_file).split(".")[0]
    output_dir = os.path.join(output_dir, f"{filename}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    device = torch.device(device)

    nii_image = nib.load(input_file)
    volume = nii_image.get_fdata()
    voxel_spacings = np.array(nii_image.header.get_zooms())
    delx = pixel_size
    drr = DRR(
        volume,
        voxel_spacings,
        source_to_detector_distance / 2.0,
        height,
        delx,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        bone_attenuation_multiplier=10.0,
        patch_size=128,
    ).to(device)
    transforms = Transforms(height)


    alpha_values = torch.linspace(alpha_min, alpha_max, steps).to(device)
    beta_values = torch.linspace(beta_min, beta_max, steps).to(device)

    metadata = {
        "params": {
            "num_alpha": steps,
            "num_beta": steps,
            "vol_path": input_file,  # Update this path as needed
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }
    
    counter = 0
    for alpha in alpha_values:
        for beta in beta_values:
            pose = convert(
                torch.tensor([[alpha.item(), beta.item(), -np.pi/2.0]]),
                (torch.tensor(volume.shape) * voxel_spacings / 2).unsqueeze(0),
                parameterization="euler_angles",
                convention="ZYX",
            ).to(device)
            img = drr(pose, bone_attenuation_multiplier=3.0)
            img = transforms(img)
            img = img.squeeze().cpu().numpy()
            
            output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imageio.imwrite(output_path, img)
            logger.info(f"Saved DRR for alpha={alpha.item():.2f}, beta={beta.item():.2f} at {output_path}")

            metadata[f"{counter:04d}"] = {
                "proj": drr.compute_projection_matrix(pose).cpu().numpy().tolist(),
                "pose": pose.matrix.cpu().numpy().tolist()
            }
            counter += 1

    metadata_path = os.path.join(output_dir, 'info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved at {metadata_path}")


@template.command()
@click.option('--input_file', default="data/CTPelvic1K/dataset6_volume/dataset6_CLINIC_0001_data.nii.gz", help='Path to the volume')
@click.option('--height', default=512, help='Image height')
@click.option('--alpha_min', default=np.pi/4.0, type=float, help='Minimum alpha angle')
@click.option('--alpha_max', default=np.pi*3.0/4.0, type=float, help='Maximum alpha angle')
@click.option('--beta_min', default=-np.pi/4.0, type=float, help='Minimum beta angle')
@click.option('--beta_max', default=np.pi/4.0, type=float, help='Maximum beta angle')
@click.option('--steps', default=4, help='Number of steps in alpha and beta ranges')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--pixel_size', default=1.4, help='Pixel size')
@click.option('--output_dir', default='./data/ctpelvic1k_templates', help='Directory to save generated DRRs')
def ctpelvic1k(input_file, height, alpha_min, alpha_max, beta_min, beta_max, steps, device, output_dir, pixel_size):
    logger.info(f"Generating DRRs for file {input_file}...")
    filename = os.path.basename(input_file).split(".")[0]
    output_dir = os.path.join(output_dir, f"{filename}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    device = torch.device(device)

    nii_image = nib.load(input_file)
    volume = nii_image.get_fdata()
    voxel_spacings = np.array(nii_image.header.get_zooms())
    delx = pixel_size
    drr = DRR(
        volume,
        voxel_spacings,
        1800 / 2.0,
        height,
        delx,
        x0=0.0,
        y0=0.0,
        reverse_x_axis=True,
        renderer="siddon",
        bone_attenuation_multiplier=10.0,
        patch_size=128,
    ).to(device)
    transforms = Transforms(height)


    alpha_values = torch.linspace(alpha_min, alpha_max, steps).to(device)
    beta_values = torch.linspace(beta_min, beta_max, steps).to(device)

    metadata = {
        "params": {
            "num_alpha": steps,
            "num_beta": steps,
            "vol_path": input_file,  # Update this path as needed
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }
    
    counter = 0
    for alpha in alpha_values:
        for beta in beta_values:
            pose = convert(
                torch.tensor([[alpha.item(), beta.item(), -np.pi/2.0]]),
                (torch.tensor(volume.shape) * voxel_spacings / 2).unsqueeze(0),
                parameterization="euler_angles",
                convention="ZYX",
            ).to(device)
            img = drr(pose, bone_attenuation_multiplier=3.0)
            img = transforms(img)
            img = img.squeeze().cpu().numpy()
            
            output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imageio.imwrite(output_path, img)
            logger.info(f"Saved DRR for alpha={alpha.item():.2f}, beta={beta.item():.2f} at {output_path}")

            metadata[f"{counter:04d}"] = {
                "proj": drr.compute_projection_matrix(pose).cpu().numpy().tolist(),
                "pose": pose.matrix.cpu().numpy().tolist()
            }
            counter += 1

    metadata_path = os.path.join(output_dir, 'info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved at {metadata_path}")

@template.command()
@click.option('--id_number', default=2, help='Specimen number (1-6)')
@click.option('--height', default=512, help='Image height')
@click.option('--alpha_min', default=np.pi/4.0, type=float, help='Minimum alpha angle')
@click.option('--alpha_max', default=np.pi*3.0/4.0, type=float, help='Maximum alpha angle')
@click.option('--beta_min', default=-np.pi/4.0, type=float, help='Minimum beta angle')
@click.option('--beta_max', default=np.pi/4.0, type=float, help='Maximum beta angle')
@click.option('--steps', default=4, help='Number of steps in alpha and beta ranges')
@click.option('--device', default='cuda', help='Device to run on (e.g., cuda or cpu)')
@click.option('--output_dir', default='./data/deepfluoro_templates', help='Directory to save generated DRRs')
def deepfluoro(id_number, height, alpha_min, alpha_max, beta_min, beta_max, steps, device, output_dir):
    output_dir = os.path.join(output_dir, f"specimen_{id_number}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    device = torch.device(device)
    
    specimen = DeepFluoroDataset(id_number, filename="data/ipcai_2020_full_res_data.h5", preprocess=True)

    delx = 1.4
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        1800 / 2.0,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        renderer="siddon",
    ).to(device)
    transforms = Transforms(height)


    alpha_values = torch.linspace(alpha_min, alpha_max, steps).to(device)
    beta_values = torch.linspace(beta_min, beta_max, steps).to(device)

    metadata = {
        "params": {
            "num_alpha": steps,
            "num_beta": steps,
            "vol_path": "data/ipcai_2020_full_res_data.h5",  # Update this path as needed
            "sensor_width": height,
            "sensor_height": height,
            "pixel_size": drr.detector.delx,
            "source_to_detector_distance": drr.detector.sdr * 2
        }
    }
    
    counter = 0
    for alpha in alpha_values:
        for beta in beta_values:
            pose = convert(
                torch.tensor([[alpha.item(), beta.item(), -np.pi/2.0]]),
                (torch.tensor(specimen.volume.shape) * specimen.spacing / 2).unsqueeze(0),
                parameterization="euler_angles",
                convention="ZYX",
            ).to(device)
            img = drr(pose, bone_attenuation_multiplier=3.0)
            img = transforms(img)
            img = img.squeeze().cpu().numpy()
            
            output_path = os.path.join(output_dir, 'images', f'{counter:04d}.png')
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            imageio.imwrite(output_path, img)
            logger.info(f"Saved DRR for alpha={alpha.item():.2f}, beta={beta.item():.2f} at {output_path}")

            metadata[f"{counter:04d}"] = {
                "proj": drr.compute_projection_matrix(pose).cpu().numpy().tolist(),
                "pose": pose.matrix.cpu().numpy().tolist()
            }
            counter += 1

    metadata_path = os.path.join(output_dir, 'info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved at {metadata_path}")

@generate.command()
@click.option('--data_dir', type=str, default='./data/deepfluoro_synthetic', help='Directory containing the dataset')
@click.option('--type', type=click.Choice(['ctpelvic1k', 'deepfluoro', 'custom'], case_sensitive=False), default='deepfluoro', help='Type of dataset')
@click.option('--split_ratio', type=float, default=0.8, help='Split ratio')
def splits(data_dir, type, split_ratio):
    # Get all top directory folders, ignore files
    folders = [f for f in glob.glob(f'{data_dir}/*') if not f.endswith('.csv')]
    logger.info(f'Found {len(folders)} folders in {data_dir}')

    if type == 'deepfluoro':
        # Leave 17-1905 for testing
        folders = [f for f in folders if not os.path.basename(f).startswith('specimen_6')]

    data_dict = {}
    for folder in folders:
        data_dict[os.path.basename(folder)] = []
        for data_index in glob.glob(f'{folder}/images/*.png'):
            data_dict[os.path.basename(folder)].append(os.path.basename(data_index).split('.')[0])

    # Split data
    train_data = {}
    val_data = {}

    for key, value in data_dict.items():
        split_index = int(len(value) * split_ratio)
        train_data[key] = value[:split_index]
        val_data[key] = value[split_index:]
    
    # Write to csv
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    os.makedirs(data_dir, exist_ok=True)
    with open(train_csv, 'w') as f:
        writer = csv.writer(f)
        for key, value in train_data.items():
            for v in value:
                writer.writerow([key, 
                                 os.path.join('images', f'{v}.png'), 
                                 os.path.join('cameras', f'{v}.json'),
                                 ])
    with open(val_csv, 'w') as f:
        writer = csv.writer(f)
        for key, value in val_data.items():
            for v in value:
                writer.writerow([key, 
                                 os.path.join('images', f'{v}.png'), 
                                 os.path.join('cameras', f'{v}.json'),
                                 ])
    logger.info(f'Wrote {len(train_data)} train data to {train_csv}')
    logger.info(f'Wrote {len(val_data)} val data to {val_csv}')

    if type == 'deepfluoro':
        # Get test data
        test_data = {}
        for folder in glob.glob(f'{data_dir}/specimen_6*'):
            test_data[os.path.basename(folder)] = []
            for data_index in glob.glob(f'{folder}/images/*.png'):
                test_data[os.path.basename(folder)].append(os.path.basename(data_index).split('.')[0])

        test_csv = os.path.join(data_dir, 'test.csv')
        with open(test_csv, 'w') as f:
            writer = csv.writer(f)
            for key, value in test_data.items():
                for v in value:
                    writer.writerow([key, 
                                    os.path.join('images', f'{v}.png'), 
                                    os.path.join('cameras', f'{v}.json'),
                                    ])
        logger.info(f'Wrote {len(test_data)} test data to {test_csv}')
