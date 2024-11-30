import glob
import json
import csv
import os

import imageio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from diffdrr.pose import RigidTransform
from rayemb.utils import sample_grid_from_mask, setup_logger

logger = setup_logger(__name__)

class SyntheticCTPelvic1KDataset(Dataset):
    def __init__(self,
                 image_size,
                 data_dir,
                 num_samples,
                 template_dir,
                 vol_dir,
                 mask_dir,
                 sample_only_visible=True,
                 sampling_distance=5,
                 num_templates=4,
                 split="train",
                 augment=False):
        self.image_size = image_size
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.vol_dir = vol_dir
        self.mask_dir = mask_dir
        self.template_dir = template_dir
        self.sample_only_visible = sample_only_visible
        self.sampling_distance = sampling_distance
        self.num_templates = num_templates
        self.split = split
        self.augment = augment
        self.templates = {}
        temp_dir = glob.glob(f'{template_dir}/*')

        with open(f'{data_dir}/{split}.csv', 'r') as f:
            reader = csv.reader(f)
            self.data = list(reader)

        for temp in temp_dir:
            temp_name = temp.split('/')[-1]
            with open(f'{temp}/info.json', 'r') as f:
                loaded_template = json.load(f)
                self.template_sensor_size_width = loaded_template['params']['sensor_width']
                self.template_sensor_size_height = loaded_template['params']['sensor_height']
                loaded_template.pop('params')
            self.templates[temp_name] = loaded_template

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.RandomChoice([
                T.RandomAdjustSharpness(0.5),
                T.RandomAutocontrast(0.5),
                T.RandomEqualize(0.5),
                T.RandomInvert(0.5),
                T.RandomSolarize(0.5),
            ]),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),  # Convert the PIL Image back to tensor
        ])

        self.no_aug_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def get_sampled_points(self, specimen_id):
        filepath_sampled_points = f'{self.data_dir}/{specimen_id}/sampled_points.npy'
        filepath_vol_landmarks = f'{self.data_dir}/{specimen_id}/vol_landmarks.npy'
        # check if the file exists
        if os.path.exists(filepath_sampled_points) and os.path.exists(filepath_vol_landmarks):
            sampled_points = np.load(filepath_sampled_points)
            vol_landmarks = np.load(filepath_vol_landmarks)
        else:
            print(f"Initial Loading volume for {specimen_id}")
            nifti_vol = nib.load(f"{self.vol_dir}/{specimen_id}.nii.gz")
            self.volume = nifti_vol.get_fdata()
            self.voxel_spacings = np.array(nifti_vol.header.get_zooms())
            maskname = "_".join(specimen_id.split("_")[:3] + ["mask", "4label"]) + ".nii.gz"
            nifti_mask = nib.load(f"{self.mask_dir}/{maskname}")
            self.mask = (nifti_mask.get_fdata() > 0).astype(np.float32)
            sampled_points = sample_grid_from_mask(self.mask, self.voxel_spacings, self.sampling_distance).astype(np.float32)
            # pick 100 points randomly from sampled_points
            vol_landmarks = sampled_points[np.random.choice(sampled_points.shape[0], 100, replace=False)]
            # save as npy
            np.save(filepath_sampled_points, sampled_points)
            np.save(filepath_vol_landmarks, vol_landmarks)
            print(f"Finished Loading volume for {specimen_id}")
        # convert to torch tensors
        return sampled_points, vol_landmarks

    def __getitem__(self, idx):
        """
        Returns:
        {
            img: (C, H, W) tensor
            templates: (T, C, H, W) tensor
            sampled_points: (N, 3) tensor
            proj_point_templates: (N, T, 2) tensor
            proj_point_img: (N, 2) tensor
            world_from_camera3d: (4, 4) tensor
            K: (3, 3) tensor
            sensor_size: (2,) tensor
            source_to_detector_distance: (1,) tensor
            pixel_size: (1,) tensor
        }
        """
        specimen_id, img_path, camera_path = self.data[idx]
        
        with open(f'{self.data_dir}/{specimen_id}/{camera_path}', 'r') as f:
            camera = json.load(f)
        img_orig = imageio.imread(f'{self.data_dir}/{specimen_id}/{img_path}')
        if self.augment:
            img = self.transforms(img_orig)
        else:
            img = self.no_aug_transforms(img_orig)

        sampled_points, vol_landmarks = self.get_sampled_points(specimen_id)
        img_sensor_size = torch.tensor([camera["params"]["sensor_width"], camera["params"]["sensor_height"]]).float()
        template_sensor_size = torch.tensor([self.template_sensor_size_width, self.template_sensor_size_height]).float()
        source_to_detector_distance = torch.tensor(camera["params"]["source_to_detector_distance"]).float()
        pixel_size = torch.tensor(camera["params"]["pixel_size"]).float()
        loaded_template = self.templates[specimen_id]

        templates = []
        proj_matrices = []
        keys = list(loaded_template.keys())
        # Set random seed to ensure reproducibility if testing
        if self.split == "test":
            np.random.seed(42)
        keys = list(loaded_template.keys())
        if self.num_templates != -1:
            keys = np.random.choice(keys, self.num_templates, replace=False)
        for k in keys:
            temp = imageio.imread(f'{self.template_dir}/{specimen_id}/images/{k}.png')
            if self.augment:
                temp = self.transforms(temp)
            else:
                temp = self.no_aug_transforms(temp)

            temp = temp.unsqueeze(0)
            proj_matrix = np.array(loaded_template[k]['proj'])
            proj_matrix = torch.tensor(proj_matrix)
            proj_matrix = proj_matrix.unsqueeze(0)
            proj_matrices.append(proj_matrix)
            templates.append(temp)
        templates = torch.cat(templates, dim=0) # (T, C, H, W)
        proj_matrices = torch.cat(proj_matrices, dim=0) # (T, 3, 4)

        # Randomly sample N points
        # sampled_points = sampled_points[np.random.choice(sampled_points.shape[0], self.num_samples, replace=False)]
        sampled_points = np.concatenate([sampled_points, np.ones((sampled_points.shape[0], 1))], axis=-1)
        sampled_points = torch.tensor(sampled_points).float() # (N, 4)

        # Now project the points using proj_matrices
        proj_point_templates = proj_matrices.float() @ sampled_points.T[None, ...]
        proj_point_templates = proj_point_templates[:,:,:].permute(2, 0, 1) # (N, T, 3)
        proj_point_templates = proj_point_templates[:, :, :2] / proj_point_templates[:, :, 2].unsqueeze(-1)
        proj_point_templates = proj_point_templates / self.template_sensor_size_width * self.image_size
        diffdrr_pose = torch.tensor(camera["pose"]).float()[0]
        query_projection_matrix = torch.tensor(camera["proj"]).float()[0]
        proj_point_img = query_projection_matrix @ sampled_points.T
        proj_point_img = proj_point_img[:2, :] / proj_point_img[2, :]
        proj_point_img = proj_point_img / img_sensor_size[0] * self.image_size
        proj_point_img = proj_point_img.T # (N, 2)

        # Now select only the visible points
        if self.sample_only_visible:
            visible_indices = (proj_point_img[:, 0] >= 0) & (proj_point_img[:, 0] < self.image_size) & (proj_point_img[:, 1] >= 0) & (proj_point_img[:, 1] < self.image_size)
            proj_point_img_ = proj_point_img[visible_indices]
            proj_point_templates_ = proj_point_templates[visible_indices]
            sampled_points_ = sampled_points[visible_indices]
            # If there are no visible points, then sample randomly
            if sampled_points_.shape[0] != 0:
                # Now select num_samples points
                if self.num_samples < sampled_points_.shape[0]:
                    idx = np.random.choice(sampled_points_.shape[0], self.num_samples, replace=False)
                    proj_point_img = proj_point_img_[idx]
                    proj_point_templates = proj_point_templates_[idx]
                    sampled_points = sampled_points_[idx]
                else:
                    idx = np.random.choice(sampled_points_.shape[0], self.num_samples, replace=True)
                    proj_point_img = proj_point_img_[idx]
                    proj_point_templates = proj_point_templates_[idx]
                    sampled_points = sampled_points_[idx]
            else:
                idx = np.random.choice(proj_point_img.shape[0], self.num_samples, replace=False)
                proj_point_img = proj_point_img[idx]
                proj_point_templates = proj_point_templates[idx]
                sampled_points = sampled_points[idx]

        else:
            idx = np.random.choice(proj_point_img.shape[0], self.num_samples, replace=False)
            proj_point_img = proj_point_img[idx]
            proj_point_templates = proj_point_templates[idx]
            sampled_points = sampled_points[idx]

        gt_vol_landmark = []
        gt_proj_landmark = []
        vol_landmarks = torch.from_numpy(vol_landmarks)
        for vol_landmark in vol_landmarks:
            gt_vol_landmark.append(vol_landmark)
            homogeneous = torch.cat([vol_landmark, torch.tensor([1.0])])
            gt_proj_landmark_s = query_projection_matrix @ homogeneous
            gt_proj_landmark_s = gt_proj_landmark_s[:2] / gt_proj_landmark_s[2]
            gt_proj_landmark_s = gt_proj_landmark_s / img_sensor_size[0] * self.image_size
            gt_proj_landmark.append(gt_proj_landmark_s)

        gt_vol_landmark = torch.stack(gt_vol_landmark, dim=0)
        gt_proj_landmark = torch.stack(gt_proj_landmark, dim=0)

        flip_xz = RigidTransform(
            torch.tensor(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        translate = RigidTransform(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, -source_to_detector_distance.item() / 2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

        # Update camera matrix
        K = torch.tensor([
            [source_to_detector_distance.item() / pixel_size.item(), 0.0, img_sensor_size[0] / 2.0],
            [0.0, source_to_detector_distance.item() / pixel_size.item(), img_sensor_size[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]).float()

        extrinsic = (
            RigidTransform(diffdrr_pose).inverse().compose(translate).compose(flip_xz)
        ).matrix[0]

        return {
            'specimen_id': specimen_id,
            'vol_dir': self.vol_dir,
            'img_id': os.path.basename(img_path).split('.')[0],
            'img': img,
            # 'img_orig': torch.from_numpy(np.array(img_orig, dtype=np.float32)),
            'templates': templates,
            'sampled_points': sampled_points,
            'projection_matrices': proj_matrices, 
            'query_projection_matrix': query_projection_matrix, # 'K @ world_from_camera3d[:3, :]
            'proj_point_templates': proj_point_templates,
            'proj_point_img': proj_point_img,
            'diffdrr_pose': diffdrr_pose,
            'extrinsic': extrinsic,
            'img_sensor_size': img_sensor_size,
            'template_sensor_size': template_sensor_size,
            'source_to_detector_distance': source_to_detector_distance,
            'pixel_size': pixel_size,
            'gt_vol_landmarks': gt_vol_landmark,
            'gt_proj_landmarks': gt_proj_landmark,
            'K': K,
            'flip_xz': flip_xz.matrix,
            'translate': translate.matrix,
        }

class SmallSyntheticCTPelvic1KDataModule(LightningDataModule):
    def __init__(self, 
                 data_dir,
                 image_size,
                 template_dir, 
                 vol_dir, 
                 mask_dir,
                 num_samples, 
                 batch_size, 
                 sampling_distance,
                 num_templates,
                 sample_only_visible,
                 num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.template_dir = template_dir
        self.vol_dir = vol_dir
        self.mask_dir = mask_dir
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.num_templates = num_templates
        self.sample_only_visible = sample_only_visible
        self.sampling_distance = sampling_distance

    def setup(self, stage=None):
        self.train_dataset = Subset(SyntheticCTPelvic1KDataset(
            data_dir=self.data_dir, 
            image_size=self.image_size,
            template_dir=self.template_dir, 
            num_samples=self.num_samples, 
            vol_dir=self.vol_dir,
            mask_dir=self.mask_dir,
            num_templates=self.num_templates,
            sampling_distance=self.sampling_distance,
            sample_only_visible=self.sample_only_visible,
            augment=True,
            split='train'), indices=range(1000))
        self.val_dataset = Subset(SyntheticCTPelvic1KDataset(
            data_dir=self.data_dir, 
            image_size=self.image_size,
            template_dir=self.template_dir, 
            vol_dir=self.vol_dir,
            mask_dir=self.mask_dir,
            num_templates=self.num_templates,
            num_samples=self.num_samples, 
            sampling_distance=self.sampling_distance,
            sample_only_visible=self.sample_only_visible,
            split='val'), indices=range(100))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class SyntheticCTPelvic1KDataModule(LightningDataModule):
    def __init__(self, 
                 data_dir,
                 image_size,
                 template_dir, 
                 vol_dir,
                 mask_dir,
                 num_samples, 
                 batch_size, 
                 sampling_distance,
                 num_templates,
                 sample_only_visible,
                 num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.template_dir = template_dir
        self.vol_dir = vol_dir
        self.mask_dir = mask_dir
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.num_templates = num_templates
        self.sample_only_visible = sample_only_visible
        self.sampling_distance = sampling_distance

    def setup(self, stage=None):
        self.train_dataset = SyntheticCTPelvic1KDataset(
            data_dir=self.data_dir, 
            image_size=self.image_size,
            template_dir=self.template_dir, 
            num_samples=self.num_samples, 
            vol_dir=self.vol_dir,
            mask_dir=self.mask_dir,
            num_templates=self.num_templates,
            sampling_distance=self.sampling_distance,
            sample_only_visible=self.sample_only_visible,
            augment=True,
            split='train')
        self.val_dataset = SyntheticCTPelvic1KDataset(
            data_dir=self.data_dir, 
            image_size=self.image_size,
            template_dir=self.template_dir, 
            vol_dir=self.vol_dir,
            mask_dir=self.mask_dir,
            num_samples=self.num_samples, 
            num_templates=self.num_templates,
            sampling_distance=self.sampling_distance,
            sample_only_visible=self.sample_only_visible,
            split='val')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)