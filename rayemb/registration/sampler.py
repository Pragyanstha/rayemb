import torch
import numpy as np
from tqdm import tqdm

from rayemb.utils import find_batch_peak_coordinates_maxval

MAX_SAMPLES = 400 # For 24GB VRAM


class RayEmbCorrespondenceSampler():
    def __init__(self, model, top_k=100, num_templates=4, num_iters=10):
        self.model = model
        self.top_k = top_k
        self.num_templates = num_templates
        self.num_iters = num_iters

    def _run_inference(self, imgs, templates, sampled_points, projection_matrices, return_features=False):
        num_iters = len(sampled_points) // MAX_SAMPLES + 1
        sims = []

        if return_features:
            query_features = []
            projected_features = []
            subspace_projections = []
        for i in range(num_iters):
            sampled_points_iter = sampled_points[i*MAX_SAMPLES: (i+1)*MAX_SAMPLES]
            if return_features:
                sims_partial, query_features_partial, projected_features_partial, subspace_projections_partial = self.model.inference(imgs, templates, sampled_points_iter, projection_matrices, return_features=True)
                query_features.append(query_features_partial.cpu())
                projected_features.append(projected_features_partial.cpu())
                subspace_projections.append(subspace_projections_partial.cpu())
            else:
                sims_partial = self.model.inference(imgs, templates, sampled_points_iter, projection_matrices)
            sims.append(sims_partial.cpu())

        sims = torch.cat(sims, dim=0)
        if return_features:
            query_features = torch.cat(query_features, dim=0)
            projected_features = torch.cat(projected_features, dim=0)
            subspace_projections = torch.cat(subspace_projections, dim=0)

        proj_point, max_vals = find_batch_peak_coordinates_maxval(sims)
        sims = sims.cpu().numpy()
        max_vals = max_vals.cpu().numpy()
        proj_point = proj_point.cpu().numpy()

        if return_features:
            return proj_point, max_vals, sims, query_features, projected_features, subspace_projections
        else:
            return proj_point, max_vals, sims

    def _random_select_templates(self, templates, projection_matrices, iteration=None):
        if iteration is not None:
            np.random.seed(iteration)
        sel_indices = np.random.choice(range(templates.shape[0]), self.num_templates, replace=False)
        return templates[sel_indices], projection_matrices[sel_indices]

    def __call__(self, imgs, templates, sampled_points, projection_matrices, return_features=False):
        # Select 4 templates
        selected_templates, selected_projection_matrices = self._random_select_templates(templates, projection_matrices, iteration=0)

        # Run initial sampling and filtering
        if return_features:
            proj_point, max_vals, sims, query_features, projected_features, subspace_projections = self._run_inference(imgs, selected_templates, sampled_points, selected_projection_matrices, return_features=return_features)
        else:
            proj_point, max_vals, sims = self._run_inference(imgs, selected_templates, sampled_points, selected_projection_matrices)
        
        # Filtering
        sel_indices = np.argsort(max_vals.copy())[::-1][:self.top_k].copy()
        proj_point = proj_point[sel_indices].astype(np.float32)
        max_vals = max_vals[sel_indices].astype(np.float32)
        sims = sims[sel_indices]

        if return_features:
            query_features = query_features[sel_indices]
            projected_features = projected_features[sel_indices]
            subspace_projections = subspace_projections[sel_indices]

        if len(sel_indices) < 5:
            sel_indices = np.arange(sampled_points.shape[0])
            print(f"Using all templates for {sel_indices.shape[0]} points")

        sampled_points = sampled_points[sel_indices]
        # Now for n iters, sample random templates
        b_sims = []
        b_proj_point = []
        b_max_vals = []
        for i in range(self.num_iters):
            # Set iteration for random seed to ensure reproducibility
            selected_templates, selected_projection_matrices = self._random_select_templates(templates, projection_matrices, iteration=i+1)
            pred_coord, maxval, sim = self._run_inference(imgs, selected_templates, sampled_points, selected_projection_matrices, return_features=False)
            b_sims.append(sim)
            b_proj_point.append(pred_coord)
            b_max_vals.append(maxval)

        b_sims = np.stack(b_sims, axis=1) # [N, B, H, W]
        b_proj_point = np.stack(b_proj_point, axis=1) # [N, B, 2]
        b_max_vals = np.stack(b_max_vals, axis=1) # [N, B]

        b_sel_indices = np.argmax(b_max_vals, axis=1) # [N]
        b_proj_point = b_proj_point[np.arange(b_proj_point.shape[0]), b_sel_indices].astype(np.float32)  # [N, 2]
        b_max_vals = b_max_vals[np.arange(b_max_vals.shape[0]), b_sel_indices].astype(np.float32) # [N]
        b_sims = b_sims[np.arange(b_sims.shape[0]), b_sel_indices] # [N, H, W]

        sampled_points = sampled_points.cpu().numpy()
        if return_features:
            return sampled_points, b_proj_point, b_max_vals, b_sims, sel_indices, query_features, projected_features, subspace_projections
        else:
            return sampled_points, b_proj_point, b_max_vals, b_sims, sel_indices


class FixedCorrespondenceSampler():
    def __init__(self, model, filter_th=0.2):
        self.model = model
        self.filter_th = filter_th

    def __call__(self, imgs):

        res = self.model.inference(imgs)

        pred_heatmaps = res["pred_heatmaps"].detach().cpu().numpy()
        proj_point = res["pred_proj_points"].detach().cpu().numpy()

        max_vals = np.max(pred_heatmaps.reshape(pred_heatmaps.shape[0], -1), axis=-1)

        # sort by max val
        sort_indices = np.where(max_vals > self.filter_th)
        proj_point = proj_point[sort_indices].astype(np.float32)
        max_vals = max_vals[sort_indices].astype(np.float32)
        pred_heatmaps = pred_heatmaps[sort_indices]

        return pred_heatmaps, proj_point, max_vals, sort_indices
