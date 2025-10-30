# VideoWeave: A Data-Centric Approach for Efficient Video Understanding

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Data Preparation**](#data-preparation) | [**Training**](#training) | [**Inference**](#inference) | [**Citation**](#citation)

VideoWeave is a data-centric recipe for efficient video understanding. It focuses on constructing informative multi-video training examples by clustering videos and composing a small number of frames per video, yielding strong temporal and semantic understanding at significantly reduced compute.

- **Video-first backbones**: Adds VideoCLIP ViT backbones that accept frame sequences and aggregate features efficiently.
- **Data-centric video construction**: Utilities to sample frames, cluster videos, and create multi-video training items from WebVid.
- **Flexible datasets**: Image-only LLaVA v1.5 reproduction and Video-LLaVA variant, plus WebVid-based training/validation splits.
- **Language models**: Works with Mistral, Llama 3, Phi-3, Gemma, and others via HF Transformers.

This repository retains upstream Prismatic functionality, with additional components for video pretraining and evaluation.

---

## Installation

Prereqs: Python ≥3.8 (tested on 3.10), PyTorch ≥2.1, Torchvision ≥0.16, Transformers ≥4.38.

```bash
pip install -e .

# Optional but recommended for training larger models
pip install packaging ninja
pip install flash-attn --no-build-isolation  # if your CUDA/PyTorch stack supports it

# For WebVid processing scripts
pip install decord tqdm pandas pillow
```

Create a file `.hf_token` in the repo root with a Hugging Face Read token if you use gated models.

---

## Data Preparation

We support both image and video-style datasets.

- LLaVA v1.5 image data: use the automated downloader.

```bash
python scripts/preprocess.py --dataset_id llava-v1.5-instruct --root_dir <DATA_ROOT>
python scripts/preprocess.py --dataset_id llava-laion-cc-sbu-558k --root_dir <DATA_ROOT>
```

- WebVid videos (frames + metadata): prepare per-video frame folders and JSON metadata following the schema below. Utilities in `scripts/webvid.py` help cluster videos and build multi-video training examples.

Example metadata schema expected by the loader (frames are relative to a frames root dir):

```json
{
  "id": "0000000000",
  "frames": ["0000000000/0000.png", "0000000000/0001.png", "0000000000/0002.png"],
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe what is happening in the video."},
    {"from": "gpt", "value": "<caption>"}
  ]
}
```

Notes:
- The project includes scripts to cluster WebVid and to compose multi-video items; see `scripts/webvid.py` and `scripts/additional-datasets`.
- Paths in `prismatic/conf/datasets.py` are examples from our environment. Please edit those paths (root dir and json/image locations) to your local setup. In particular, update entries for `WEBVID` and `WEBVID_VAL` and for `VIDEO_LLAVA_V15` if you use the Video-LLaVA variant.

---

## Training

The entry point is `scripts/pretrain.py` (Draccus-configured). Set `--dataset.type` and `--model.type`, and override individual fields on the CLI. For video, use a `video-clip-*` vision backbone and set `--model.num_frames` to the number of frames per sample.

Single-node example (video training on WebVid):

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
  --dataset.type "webvid" \
  --model.type "one-stage+7b" \
  --model.model_id "videoweave-demo" \
  --model.llm_backbone_id "phi-3-instruct-4b" \
  --model.vision_backbone_id "video-clip-vit-b" \
  --model.image_resize_strategy "letterbox" \
  --model.num_frames 4 \
  --model.finetune_global_batch_size 1 \
  --model.finetune_per_device_batch_size 1 \
  --wandb_entity "<your_wandb_entity>" \
  --wandb_project "videoweave"
```

Video-LLaVA v1.5 style fine-tuning (if you’ve created the video-form LLaVA split):

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --dataset.type "video-llava-v15" \
  --model.type "one-stage+7b" \
  --model.llm_backbone_id "mistral-v0.2-7b-instruct" \
  --model.vision_backbone_id "video-clip-vit-l-336px" \
  --model.image_resize_strategy "letterbox" \
  --model.num_frames 8 \
  --wandb_entity "<your_wandb_entity>" \
  --wandb_project "videoweave"
```

Tips:
- `.hf_token` is used automatically if you specify gated LMs.

---

## Inference

You can load a trained run directory or a HF Hub model and generate from a sequence of frames (list of PIL images). Example:

```python
import torch
from PIL import Image
from pathlib import Path
from prismatic import load

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load from a local training run directory (contains config.json and checkpoints/)
vlm = load("/path/to/run_dir", hf_token=hf_token, num_frames=4)
vlm.to(device, dtype=torch.bfloat16)

# Prepare a list of frames (PIL Images)
frames = [Image.open(f"/path/to/vid/000{i}.png").convert("RGB") for i in range(4)]

prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message="Describe what is happening in the video.")
prompt_text = prompt_builder.get_prompt()

text = vlm.generate(frames, prompt_text, do_sample=False, temperature=0.2, max_new_tokens=256)
print(text)
```

---

## Repository Structure

- `prismatic/` – core library: configs, backbones (vision + LLM), VLM wrapper, training strategies, preprocessing
- `scripts/` – dataset processing, training, generation; includes `webvid.py` helpers for video data
- `experiment_scripts/` – shell scripts for various controlled experiments
- `run_pretrain.sh` – example launcher for a simple video pretraining run

---

## Citation

If you find this repository useful, please cite the VideoWeave paper and the Prismatic VLMs work.

VideoWeave:

```bibtex
@article{videoweave2025,
  title={VideoWeave: A Data-Centric Approach for Efficient Video Understanding},
  author={Author(s)},
  year={2025},
  journal={},
}
```

Prismatic VLMs:

```bibtex
@inproceedings{karamcheti2024prismatic,
  title = {Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models},
  author = {Siddharth Karamcheti and Suraj Nair and Ashwin Balakrishna and Percy Liang and Thomas Kollar and Dorsa Sadigh},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024},
}
```

---

## License

This codebase is released under the MIT License (see `LICENSE`). Models trained using gated LMs or third-party datasets may inherit usage restrictions from their respective licenses/terms.
