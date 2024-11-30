#!/bin/bash
CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

specimens=("1" "2" "3" "4" "5" "6")
# Loop over each volume name and run the command
for specimen in "${specimens[@]}"; do
    echo "Processing volume: $specimen"

    rayemb generate dataset deepfluoro --id_number $specimen \
    --output_dir data/deepfluoro_synthetic \
    --height 512 --num_samples 10000
done

echo "Now create train/val/test split"
rayemb generate split --data_dir data/deepfluoro_synthetic --type deepfluoro --split_ratio 0.8