#!/bin/bash
CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

input_dir="data/CTPelvic1K/dataset6_volume"  # Specify the directory containing the files

# Loop over each file in the directory
for input_file in "$input_dir"/*; do
    echo "Processing file: $input_file"

    # Set default distance
    pixel_size=1.4

    rayemb generate template ctpelvic1k --input_file $input_file \
    --output_dir data/ctpelvic1k_templates \
    --height 512 --steps 18
done