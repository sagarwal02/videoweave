# '''
# total input frames = 16, videos per cluster = 8
# CLIP-ViT@336 + llama-2
# '''
export OMP_NUM_THREADS=16
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=0   # disables timeout (safe for testing/debugging)
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=86400  # 24 hours in seconds

torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "prism-clip+7b-webvid-train-45k-total-input-frames=16-videos-per-cluster=8-epochs=5-gpus=2-004" \
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
  --model.finetune_epochs 1 \
  --dataset.type "webvid" \
  --dataset.finetune_stage_components '["/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_train_45k_videos_per_cluster=8_total_input_frames=16.json", "/vision/u/silsingh/prismatic-vlms/webvid_num_frames=2"]' \
  --val_dataset.type "webvid_val" \
  --val_dataset.finetune_stage_components '["/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_val_5k_videos_per_cluster=8_total_input_frames=16.json", "/vision/u/silsingh/prismatic-vlms/webvid_num_frames=2"]' \
  --val_dataset.eval_interval 10 \
  --val_dataset.lmms_eval_list '["videomme"]'
  # Eval every 10 steps