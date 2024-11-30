#!/bin/bash
CUDA_VISIBLE_DEVICES=0  # Set the GPU ID

input_dir="data/CTPelvic1K/dataset6_volume"  # Specify the directory containing the files

# Loop over each file in the directory
for input_file in "$input_dir"/*; do
    echo "Processing file: $input_file"

    rayemb generate dataset ctpelvic1k --input_file "$input_file" \
    --output_dir data/ctpelvic1k_synthetic \
    --height 512 --num_samples 600
done

echo "Now create train/val split (no test split since we don't need to evaluate on this dataset)"
rayemb generate split --data_dir data/ctpelvic1k_synthetic --type ctpelvic1k --split_ratio 0.8