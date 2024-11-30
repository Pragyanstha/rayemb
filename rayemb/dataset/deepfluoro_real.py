import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import glob
import json
import imageio
import numpy as np

from diffdrr.pose import RigidTransform
from .deepfluoro import DeepFluoroDataset 
from rayemb.utils import sample_grid_volume_from_ct, Transforms


class RealDeepFluoroDataset(Dataset):
    def __init__(
            self, 
            specimen_id: str,
            image_size,
            num_samples,
            template_dir,
            h5_dir,
            sample_only_visible=True,
            sampling_distance=5,
            num_templates=4,
            augment=False
            ):
        # check if specimen_id is valid
        if specimen_id not in ["specimen_1", "specimen_2", "specimen_3", "specimen_4", "specimen_5", "specimen_6"]:
            raise ValueError(f"Invalid specimen_id: {specimen_id}. Valid specimen_ids are 'specimen_1', 'specimen_2', 'specimen_3', 'specimen_4', 'specimen_5', 'specimen_6'")
        
        self.specimen_id = specimen_id
        self.image_size = image_size
        self.num_samples = num_samples
        self.template_dir = template_dir
        self.sample_only_visible = sample_only_visible
        self.augment = augment
        self.h5_dir = h5_dir
        self.num_templates = num_templates
        self.templates = {}
        temp_dir = glob.glob(f'{template_dir}/*')

        self.specimen = DeepFluoroDataset(id_number=int(specimen_id.split('_')[-1]), filename=h5_dir, preprocess=True)
        self.sampled_points = sample_grid_volume_from_ct(self.specimen.volume, self.specimen.spacing, resolution=sampling_distance)
        
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
            T.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ])

        self.no_aug_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.Grayscale(num_output_channels=3),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.specimen.projections)

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
        vol_landmarks = self.specimen.fiducials[0]

        img_orig, gt_pose, gt_proj_landmark_deepfluoro = self.specimen[idx]
        image_idx = f"{idx:04d}"

        if self.augment:
            img = self.transforms(img_orig.squeeze().numpy())
        else:
            img = self.no_aug_transforms(img_orig.squeeze().numpy())

        img_sensor_size = torch.tensor([self.specimen.num_cols, self.specimen.num_rows]).float()
        template_sensor_size = torch.tensor([self.template_sensor_size_width, self.template_sensor_size_height]).float()
        source_to_detector_distance = torch.tensor(self.specimen.focal_len).float()
        pixel_size = torch.tensor(self.specimen.proj_col_spacing).float()
        loaded_template = self.templates[self.specimen_id]

        templates = []
        proj_matrices = []
        # select num_templates from loaded_template
        # Set random seed to ensure reproducibility
        np.random.seed(42)
        keys = list(loaded_template.keys())
        if self.num_templates != -1:
            keys = np.random.choice(keys, self.num_templates, replace=False)
        for k in keys:
            temp = imageio.imread(f'{self.template_dir}/{self.specimen_id}/images/{k}.png')
            temp = self.no_aug_transforms(temp)
            temp = temp.unsqueeze(0)
            proj_matrix = np.array(loaded_template[k]['proj'])
            proj_matrix = torch.tensor(proj_matrix)
            proj_matrix = proj_matrix.unsqueeze(0)
            proj_matrices.append(proj_matrix)
            templates.append(temp)
        templates = torch.cat(templates, dim=0) # (T, C, H, W)
        proj_matrices = torch.cat(proj_matrices, dim=0) # (T, 3, 4)

        sampled_points = self.sampled_points
        sampled_points = np.concatenate([sampled_points, np.ones((sampled_points.shape[0], 1))], axis=-1)
        sampled_points = torch.tensor(sampled_points).float() # (N, 4)

        K = self.specimen.intrinsic.clone().detach().float()

        # Now project the points using proj_matrices
        proj_point_templates = proj_matrices.float() @ sampled_points.T[None, ...]
        proj_point_templates = proj_point_templates[:,:,:].permute(2, 0, 1) # (N, T, 3)
        proj_point_templates = proj_point_templates[:, :, :2] / proj_point_templates[:, :, 2].unsqueeze(-1)
        proj_point_templates = proj_point_templates / self.template_sensor_size_width * self.image_size
        diffdrr_pose = gt_pose.matrix[0]
        query_projection_matrix = gt_pose.inverse().compose(self.specimen.translate).compose(self.specimen.flip_xz).matrix
        query_projection_matrix = K @ query_projection_matrix[0][:-1, :]
        proj_point_img = query_projection_matrix @ sampled_points.T
        proj_point_img = proj_point_img[:2, :] / proj_point_img[2, :] - 50.0
        proj_point_img = proj_point_img / (img_sensor_size[0]-100) * self.image_size
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
        for vol_landmark in vol_landmarks:
            gt_vol_landmark.append(vol_landmark)
            homogeneous = torch.cat([vol_landmark, torch.tensor([1.0])])
            gt_proj_landmark_s = query_projection_matrix @ homogeneous
            gt_proj_landmark_s = gt_proj_landmark_s[:2] / gt_proj_landmark_s[2] - 50.0
            gt_proj_landmark_s = gt_proj_landmark_s / (img_sensor_size[0]-100) * self.image_size
            gt_proj_landmark.append(gt_proj_landmark_s)

        gt_vol_landmark = torch.stack(gt_vol_landmark, dim=0)
        gt_proj_landmark = torch.stack(gt_proj_landmark, dim=0)
        # flip xy
        # Flip the x and y coordinates for visualization
        gt_proj_landmark_deepfluoro = img_sensor_size - gt_proj_landmark_deepfluoro - 50.0
        gt_proj_landmark_deepfluoro = gt_proj_landmark_deepfluoro / (img_sensor_size[0]-100) * self.image_size

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

        extrinsic = (
            RigidTransform(diffdrr_pose).inverse().compose(translate).compose(flip_xz)
        ).matrix[0]

        # make K positive
        K = K.abs()

        return {
            'specimen_id': self.specimen_id,
            'img_id': image_idx,
            'img': img,
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
