#!/bin/bash
CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

specimens=("1" "2" "3" "4" "5" "6")
# Loop over each volume name and run the command
for specimen in "${specimens[@]}"; do
    echo "Processing volume: $specimen"

    # Set default distance
    pixel_size=1.4

    rayemb generate template deepfluoro --id_number $specimen \
    --output_dir data/deepfluoro_templates \
    --height 512 --steps 18
done