#!/bin/bash

name="qwen2_5vl_7b_full_sft_mark_32_3D_img128_SpaceSpan"
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}

echo "Calculating scores using $CHUNKS GPUs"

ARGS=(
    "./evaluate/configs/eval_configs.py"
    "evaluate" "True"
    "model" "qwenvl"
    "batch_size" "4"
    "num_workers" "16"
    "model_path" "./outputs/vlnce/${name}"
    "scene_anno" "./evaluate/annotation/selected_images_mark_3D_val_32.json"
    "save_interval" "2"
    "val_tag" "scanqa"
    "output_dir" "eval_outputs/outputs_3D_mark/${name}/real_test"
    "num_chunks" "$CHUNKS"
    "calculate_score_tag" "scanqa"
)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python evaluate/calculate_scores.py "${ARGS[@]}"
