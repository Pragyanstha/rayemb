<div align="center">    
 
# RayEmb 
Arbitrary Landmark Detection in X-Ray Images Using Ray Embedding Subspace   

[![Paper](http://img.shields.io/badge/cs.CV-2410.08152-B31B1B.svg)](https://arxiv.org/abs/2410.08152)
[![Conference](http://img.shields.io/badge/ACCV-2024-4b44ce.svg)](https://accv2024.org/)
[![Project](https://img.shields.io/badge/project_page-rayemb-blue.svg)](https://pragyanstha.github.io/rayemb/)

<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
 -->

<!--  
Conference   
-->   
</div>

![teaser](./assets/concept.png)
Comparison of landmark detection results between conventional fixed landmark
estimation and our arbitrary landmark estimation method. The 3D landmarks are
shown in magenta on the left, while the estimated 2D landmarks are displayed in
cyan and the ground truth in magenta on the right. Our method can generate a large
number of corresponding pairs of 3D landmarks and 2D projections, whereas the fixed
landmark estimation approach is limited to the pre-annotated landmarks.

## 🔥 Updates
- ```2024/12/09``` : [Interactive demo](https://pragyanstha.github.io/rayemb/demo) is live!
- ```2024/12/04``` : [Project page](https://pragyanstha.github.io/rayemb/) is live!
- ```2024/12/01``` : Code available.

## ⭐ Overview

RayEmb introduces a novel approach for detecting arbitrary landmarks in X-ray images using ray embedding subspace. Our approach represents 3D points as distinct subspaces, formed by feature vectors (referred to as ray embeddings) corresponding to intersecting rays.
Establishing 2D-3D correspondences then becomes a task of finding ray embeddings that are close to a given subspace, essentially performing an intersection test.  

## 🚀 Features

- A CLI for downloading data, preprocessing, training and evaluating models.
- A PyTorch implementation of the RayEmb and FixedLandmark models.
- [OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc9317d6a9) based PnP + RANSAC based 2D-3D registration for initial pose estimates
- [DiffDRR](https://github.com/eigenvivek/DiffDRR) based refinement module.

## 📚 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
We have tested the code on a RTX 3090 and an H100 GPU.

## 🛠️ Installation and Setup
Install the dependencies using poetry.
```bash
poetry install
```
Check that rayemb-cli is in your path and is executable.
```bash
rayemb --help
```
```
Usage: rayemb [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  evaluate
  generate
  train
```

Download the original DeepFluoro dataset (only the real x-raysfor testing) using the following command:
```bash
rayemb download deepfluoro \
--data_dir ./data
```
Preprocess the DeepFluoro dataset to get the templates using the following command:
```bash
sh scripts/generate/template_deepfluoro.sh
```
Download the RayEmb-CTPelvic1k checkpoint using the following command:
```bash
rayemb download checkpoint \
--checkpoint_dir ./checkpoints \
--model rayemb \
--dataset ctpelvic1k
```
This can be used to evaluate the model on the CTPelvic1k dataset as well as DeepFluoro dataset.

## 🎖️ Evaluation
```bash
rayemb evaluate arbitrary-landmark deepfluoro \
--checkpoint_path ./checkpoints/rayemb-ctpelvic1k.ckpt \
--num_templates 4 \
--image_size 224 \
--template_dir ./data/deepfluoro_templates \
--data_dir ./data/ipcai_2020_full_res_data.h5
```

## 📔 Training and Testing Custom Data
Below is an example for generating templates and training data from a single Nifti file. If you want to generate templates and images for multiple Nifti files, 
you can write a shell script to loop through the files and generate templates and images, please refer to the [generate](./scripts/generate/template_ctpelvic1k.sh) command for more details.

```bash
rayemb generate template custom \
--input_file <path_to_nifti_file> \
--output_dir <path_to_output_templates_dir> \
--height <image_height> \
--steps <number_of_steps> \
--pixel_size <pixel_size> \
--source_to_detector_distance <source_to_detector_distance>
```

Now, we can generate the training and testing images using the following command:
```bash
rayemb generate dataset custom \
--input_file <path_to_nifti_file> \
--mask_dir <path_to_mask_dir> \ # optional, if not provided, a mask will be generated using the threshold
--threshold <threshold> \ # only used if mask_dir is not provided
--output_dir <path_to_output_images_dir> \
--height <image_height> \
--num_samples <number_of_samples> \
--device <device> \
--source_to_detector_distance <source_to_detector_distance> \
--pixel_size <pixel_size> \
```

Split the generated images into training and validation sets using the following command:
```bash
rayemb generate splits \
--data_dir <path_to_data_dir> \
--type custom \
--split_ratio <split_ratio>
```

Train the model using the following command:
```bash
sh scripts/train/rayemb_custom.sh <path_to_data_dir> <path_to_template_dir>
```

## 🏷️ TODO
- [x] Update the readme for evaluation, synthetic data generation and template generation.
- [x] Update the readme for training.
- [ ] Add typing and annotations to the codebase.

## Contact

For any questions or collaboration inquiries, please contact shrestha.pragyan@image.iit.tsukuba.ac.jp.

## Acknowledgements

We would like to thank [eigenvivek](https://github.com/eigenvivek) for his awesome [DiffDRR](https://github.com/eigenvivek/DiffDRR) codebase.

## Citations
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```
@InProceedings{Shrestha_2024_ACCV,
    author    = {Shrestha, Pragyan and Xie, Chun and Yoshii, Yuichi and Kitahara, Itaru},
    title     = {RayEmb: Arbitrary Landmark Detection in X-Ray Images Using Ray Embedding Subspace},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {665-681}
}
```
