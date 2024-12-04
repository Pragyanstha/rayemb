#!/bin/bash
input_dir=$1
output_dir=$2
heatmap=$input_dir/heatmap.gif
trajectory=$input_dir/trajectory.gif
palette=$output_dir/palette.png
teaser=$output_dir/teaser.gif
ffmpeg -i $trajectory -i $heatmap -filter_complex "[0:v][1:v]hstack=inputs=2, palettegen" $palette
ffmpeg -i $trajectory -i $heatmap -i $palette -filter_complex "[0:v][1:v]hstack=inputs=2, paletteuse" $teaser