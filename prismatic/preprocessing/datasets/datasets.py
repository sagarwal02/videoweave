"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type
import os
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CodeGenTokenizerFast, GemmaTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase, PreTrainedTokenizerFast, LlamaTokenizer

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
import random 
import decord

# HuggingFace Default / Llama-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def convert_to_prismatic_format(conversation_data):
    """
    Convert conversation data from the given format:
    {
        'id': 'train-0000000000',
        'source': 'webvid10m',
        'conversations': [{'images': ['0/0.png', '0/1.png', '0/2.png', '0/3.png', '0/4.png', '0/5.png', '0/6.png', '0/7.png'], 'user': 'Describe what is happening in the video.', 'assistant': 'Aerial shot winter forest'}]
    }
    
    To the desired format:
    {
        'id': '0000000000',
        'frames': ['0/0.png', '0/1.png', '0/2.png', '0/3.png', '0/4.png', '0/5.png', '0/6.png', '0/7.png'],
        'conversations': [
            {'from': 'human', 'value': '<image>\nDescribe what is happening in the video.'},
            {'from': 'gpt', 'value': 'Aerial shot winter forest'}
        ]
    }
    """
    # Extract the ID without 'train-' prefix
    new_id = conversation_data['id'].replace('train-', '')
    
    # Get the image list
    frames = conversation_data['conversations'][0]['images']
    
    # Build the conversation list in the new format
    new_conversations = [
        {
            'from': 'human',
            'value': f"<image>\n{conversation_data['conversations'][0]['user']}"
        },
        {
            'from': 'gpt',
            'value': conversation_data['conversations'][0]['assistant']
        }
    ]
    
    # Construct the new format
    new_format = {
        'id': new_id,
        'frames': frames,
        'conversations': new_conversations
    }
    
    return new_format


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example or "frames" in example or "videos" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        shuffle_frames: bool = False,
        eval_interval: int = None,
        lmms_eval_list: List[str] = None,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.shuffle_frames = shuffle_frames
        self.eval_interval = eval_interval
        self.lmms_eval_list = lmms_eval_list
        
        if self.shuffle_frames:
            print("Shuffling frame order for finetune dataset!")

        # Load Instruct JSON
        if self.instruct_json.suffix.endswith(".jsonl"):
            with open(self.instruct_json, "r") as f:
                self.examples = [convert_to_prismatic_format(json.loads(line)) for line in f]
        else: 
            with open(self.instruct_json, "r") as f:
                self.examples = json.load(f)


    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """

        if idx >= len(self.examples):
            return {
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([]),
                "pixel_values": torch.tensor([]),
            }
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama 1 & 2 Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            
            elif isinstance(self.tokenizer, LlamaTokenizer): # the slow version of LlamaTokenizer for Mistral
                msg = msg.rstrip()

            # Llama 3 Tokenizer (Fast).
            elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
                # Sidd said that there was no harm in always applying rstrip, so do this to be safe!
                msg = msg.rstrip()

            # Gemma Tokenizer
            elif isinstance(self.tokenizer, GemmaTokenizerFast):
                # Sidd said that there was no harm in always applying rstrip, so do this to be safe!
                msg = msg.rstrip()

            # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass

            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]
        
        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB")) # torch.size([3, 224, 224])

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        # === Handle multi-image (video) inputs ===
        elif "frames" in self.examples[idx]:
            image_paths = [Path(image_path) for image_path in self.examples[idx]["frames"]]

            if self.shuffle_frames:
                random.shuffle(image_paths)

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            pixel_values = [self.image_transform(Image.open(self.image_dir / image_path).convert("RGB")) for image_path in image_paths]
            
            # stack the pixel values to change from list of [3, 224, 224] to [num_frames, 3, 224, 224]
            if isinstance(pixel_values[0], torch.Tensor):
                input_data = torch.stack(pixel_values, dim=0).to(pixel_values[0].device) # stack and put to device of first tensor
            elif isinstance(pixel_values[0], Dict):
                keys = pixel_values[0].keys()
                input_data = dict()
                for k in keys:
                    t = [el[k] for el in pixel_values]
                    t = torch.stack(t, dim=0).to(t[0].device)
                    input_data[k] = t
            else:
                raise Exception(f"unsupported pixel values: {type(pixel_values[0])}")
            
            return dict(pixel_values=input_data, input_ids=input_ids, labels=labels)

        elif "videos" in self.examples[idx]:
            try:
                frames_per_video = self.examples[idx]["num_frames_per_video"]
            except KeyError:
                print("Should specify frames per video in the example! Otherwise defaulting to 16.")
                frames_per_video = 16

            video_paths = [Path(video_path) for video_path in self.examples[idx]["videos"]]

            if self.shuffle_frames:
                random.shuffle(video_paths)

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Number of frames per video to sample using decord and transform with self.image_transform
            # Use decord to sample 'frames_per_video' frames from each video uniformly randomly during training. Ensure order is preserved.
            # If the video is shorter than 'frames_per_video', sample all frames and repeat final frame if necessary.

            input_frames = []

            for video_path in video_paths:
                # Load video with decord to find length, sample frames from 0 to length - 1 uniformly, sort and get RGB vals
                try:
                    video_reader = decord.VideoReader(str(video_path))
                    num_total_frames = len(video_reader)
                    frame_indices = np.random.choice(num_total_frames, size=frames_per_video, replace=False)
                    if not self.shuffle_frames: 
                        frame_indices = np.sort(frame_indices)
                    decord_vals = video_reader.get_batch(frame_indices).asnumpy()
                except Exception as e:
                    print("Error loading video:", str(video_path), ":", e)
                    decord_vals = np.zeros((frames_per_video, 224, 224, 3), dtype=np.uint8) # default to black image if cannot load video
                frames = [self.image_transform(Image.fromarray(decord_val)) for decord_val in decord_vals]
                input_frames.extend(frames)

            input_data = torch.stack(input_frames).to(input_frames[0].device)
                        
            return dict(pixel_values=input_data, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example or "frames" in example or "videos" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
    