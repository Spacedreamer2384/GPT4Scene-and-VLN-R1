#!/bin/bash
set -x -e
export NCCL_SOCKET_IFNAME=enp1s0f0
export NCCL_DEBUG=INFO

name="qwen2vl_7b_full_sft_mark_32_3D_img512_merged_aware_qwen25"

export PYTHONPATH=.
export NNODES=1
export num_gpus=4
export WANDB_DISABLED=true
export full_batch_size=16
export batch_size=4
export gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus*$NNODES)]

export MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
export MASTER_PORT=$((RANDOM % 101 + 29400))

export output_dir=outputs/vlnce/${name}/
export model_name_or_path=ckpts/Qwen2.5-VL-7B-Instruct
export tokenized_path=tokenizers/${name}/
export CUDA_VISIBLE_DEVICES=4,5,6,7

bash -c 'torchrun \
    --nnodes $NNODES \
    --nproc_per_node ${num_gpus:-1} \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src_qwen25/train.py \
    --tokenized_path $tokenized_path \
    --model_name_or_path $model_name_or_path \
    --do_train true \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --finetuning_type full \
    --dataset merged \
    --image_max_pixels=$((512 * 512)) \
    --image_min_pixels=$((28 * 28)) \
    --template qwen2_vl \
    --cutoff_len 32768 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir $output_dir \
    --num_train_epochs 15 \
    --logging_steps 1 \
    --save_steps 4000 \
    --save_total_limit 1 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --report_to none'