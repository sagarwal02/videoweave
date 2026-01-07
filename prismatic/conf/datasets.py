"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    eval_interval: int = None                       # Number of steps between evaluations (default: None, eval every epoch instead)
    lmms_eval_list: list[str] = None                # List of evals from lmms_eval to evaluate on
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class Video_LLaVa_V15_Config(LLaVa_V15_Config):
    dataset_id: str = "video-llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k_videos_2frame.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")

@dataclass
class WebVid(DatasetConfig):
    dataset_id: str = "webvid"

    align_stage_components: Tuple[Path, Path] = (
        Path("/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_train_45k_cluster_size=4.json"),
        Path("/vision/u/silsingh/prismatic-vlms/webvid")
    )

    finetune_stage_components: Tuple[Path, Path] = (
        Path("/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_train_45k_cluster_size=4.json"),
        Path("/vision/u/silsingh/prismatic-vlms/webvid_num_frames=1")
    )

    dataset_root_dir: Path = Path("/vision/u/silsingh/prismatic-vlms")


@dataclass
class WebVid_Val(DatasetConfig):
    dataset_id: str = "webvid_val"

    finetune_stage_components: Tuple[Path, Path] = (
        Path("/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_val_5k_cluster_size=4.json"),
        Path("/vision/u/silsingh/prismatic-vlms/webvid_num_frames=1")
    )

    align_stage_components: Tuple[Path, Path] = (
        Path("/vision/u/silsingh/prismatic-vlms/dataset_splits/webvid_val_5k_cluster_size=4.json"),
        Path("/vision/u/silsingh/prismatic-vlms/webvid_cluster_size=4")
    )

    dataset_root_dir: Path = Path("/vision/u/silsingh/prismatic-vlms")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    # === Video Datasets ===
    VIDEO_LLAVA_V15 = Video_LLaVa_V15_Config
    WEBVID = WebVid
    WEBVID_VAL = WebVid_Val

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
