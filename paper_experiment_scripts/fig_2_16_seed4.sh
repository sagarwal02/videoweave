#!/bin/bash
#SBATCH --job-name=fig_1_16_seed4
#SBATCH --partition=simurgh
#SBATCH --mem=128GB
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --account=simurgh
#SBATCH --time=10-00:00:00
#SBATCH --output=slurm-%j.out

export OMP_NUM_THREADS=16

# 160k total videos, 80k steps (bs = 2)

echo "Running fig_1_16_seed4"

# Saves every 8k steps
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "prism-clip+7b-webvid-train-160k-total-input-frames=16-videos-per-cluster=16-epochs=1-gpus=2-seed4" \
  --seed 4 \
  --dataset.type "webvid" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "video-clip-vit-l-336px" \
  --model.arch_specifier "no-align+gelu-mlp" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 1 \
  --model.num_frames 16 \
  --model.rope_scaling_factor 3.0 \
  --model.init_from_model "prism-clip+7b" \
  --model.repo_id "TRI-ML/prismatic-vlms" \
  --model.ckpt_interval 500 \
  --model.finetune_epochs 1 \
  --dataset.finetune_stage_components '["/vision/u/zanedurante/prismatic-vlms/dataset_splits/webvid_160k_video_split.json_video_16_frames_1.json", ""]' \
