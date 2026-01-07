"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling
from prismatic.preprocessing.datasets.datasets import convert_to_prismatic_format
import json
import wandb
import subprocess
import datetime
import time

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def set_trace():
    if dist.get_rank() == 0:
        import pdb; pdb.set_trace()

# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        dry_run: bool = False,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        self.dry_run = dry_run

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        use_idx: bool = False,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        val_dataset: None, # optionally include validation dataset per epoch
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            if isinstance(dataset, list):
                modality_lengths = dataset[0].get_modality_lengths()
                sampler = []
                for dset in dataset:
                    subset_sampler = SplitModalitySampler(
                        dset,
                        modality_lengths,
                        global_batch_size=self.global_batch_size,
                        num_replicas=overwatch.world_size(),
                        rank=overwatch.rank(),
                        seed=seed,
                        drop_last=False,
                    )
                    sampler.append(subset_sampler)
            else:
                modality_lengths = dataset.get_modality_lengths()
                sampler = SplitModalitySampler(
                    dataset,
                    modality_lengths,
                    global_batch_size=self.global_batch_size,
                    num_replicas=overwatch.world_size(),
                    rank=overwatch.rank(),
                    seed=seed,
                    drop_last=False,
                )
            val_modality_lengths = val_dataset.get_modality_lengths()
            val_sampler = SplitModalitySampler(
                val_dataset,
                val_modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            if isinstance(dataset, list):
                sampler = []
                for dset in dataset:
                    subset_sampler = DistributedSampler(
                        dset,
                        num_replicas=overwatch.world_size(),
                        rank=overwatch.rank(),
                        shuffle=True,
                        seed=seed,
                        drop_last=False,
                    )
                    sampler.append(subset_sampler)
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=overwatch.world_size(),
                    rank=overwatch.rank(),
                    shuffle=True,
                    seed=seed,
                    drop_last=False,
                )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        if isinstance(dataset, list):
            dataloader = []
            for dset,smplr in zip(dataset, sampler):
                dloader = DataLoader(
                    dset,
                    batch_size=self.per_device_batch_size,
                    sampler=smplr,
                    collate_fn=collator,
                    num_workers=2,
                    worker_init_fn=self.worker_init_fn,
                )
                dataloader.append(dloader)

        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.per_device_batch_size,
                sampler=sampler,
                collate_fn=collator,
                num_workers=2,
                worker_init_fn=self.worker_init_fn,
            )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.per_device_batch_size,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )


        # Max Steps vs. Epochs Computation
        if isinstance(dataset, list):
            len_dataloader = len(dataloader[0])
        else:
            len_dataloader = len(dataloader)
        steps_per_epoch = len_dataloader // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        # print('self.max_steps = ', self.max_steps, 'metrics.global_step = ', metrics.global_step)
        # import pdb; pdb.set_trace()
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len_dataloader // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                if "schedule_free" in self.lr_scheduler_type:
                    self.optimizer.train()
                if isinstance(dataset, list):
                    sampler[epoch].set_epoch(epoch)
                    dloader = dataloader[epoch]
                else:
                    sampler.set_epoch(epoch)
                    dloader = dataloader

                # Zero-Gradients (just in case)
                if "schedule_free" not in self.lr_scheduler_type:
                    self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        if "schedule_free" not in self.lr_scheduler_type:
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()

                        # Push Metrics
                        if "schedule_free" not in self.lr_scheduler_type:
                            metrics.commit(lr=self.lr_scheduler.get_last_lr()[0])
                        metrics.commit(global_step=metrics.global_step + 1)
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            dist.barrier()
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())

                            return

                        # Update Progress Bar
                    
                        progress.update()
                        progress.set_description(status)

                        #set_trace() 

                        # Save checkpoint every ckpt_interval steps
                        if hasattr(self.vlm, 'ckpt_interval') and self.vlm.ckpt_interval is not None:
                            if (train_idx + 1) % self.vlm.ckpt_interval == 0:
                                dist.barrier()
                                overwatch.info(f'Saving checkpoint.. step = {metrics.global_step}, loss = {loss.item()}')
                                if "schedule_free" in self.lr_scheduler_type:
                                    self.vlm.eval()
                                    self.optimizer.eval()

                                self.save_checkpoint(metrics.run_dir, metrics.global_step, train_idx + 1, None, use_idx=True)
                                if "schedule_free" in self.lr_scheduler_type:
                                    self.vlm.train()
                                    self.optimizer.train()
                        
                # Do eval at end of each epoch if `eval_interval` is not set
                if not hasattr(val_dataset, 'eval_interval') or val_dataset.eval_interval is None:
                    self.do_validation(val_dataset, val_dataloader)
                
                # Save checkpoint at end of each epoch (if `self.max_steps` is None)
                if self.max_steps is None: # and overwatch.is_rank_zero()
                    overwatch.info(f'Saving checkpoint.. epoch = {epoch+1}, loss = {loss.item()}')
                    dist.barrier()  # to avoid timeout errors during model saving as per https://github.com/TRI-ML/prismatic-vlms/pull/38.
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch+1, loss.item())

            # at the end of all epochs.
            # dist.barrier()
    def do_validation(self, val_dataset, val_dataloader):
        return # Just skip this for now
        # Check if current vlm state is train or eval
        orginally_train = self.vlm.training
        self.vlm.eval()
        if val_dataset is not None:
            # TODO: [Zane] Implement evals during training to make plots easier. Right now, timeout error due to using 1 GPU only I think
            #if val_dataset.lmms_eval_list is not None:
            #    self.do_lmms_eval(lmms_eval_list=val_dataset.lmms_eval_list)

            self.vlm.eval()
            val_losses = []
            #import pdb; pdb.set_trace()
            overwatch.info("Validating model..")
            for val_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                with torch.no_grad():
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss
                        val_losses.append(loss.item())
            
            if overwatch.is_rank_zero():
                overwatch.info(f'Validation loss: {sum(val_losses) / len(val_losses)}')
                wandb.log({"val_loss": sum(val_losses) / len(val_losses)})
        if orginally_train:
            self.vlm.train()
    
    def do_lmms_eval(self, lmms_eval_list):
        print('Running lmms_eval_list = ', lmms_eval_list)

        # Run evaluation
        self.vlm.eval()

        # Run eval on single GPU only, set other GPUs to wait for completion
        if dist.get_rank() == 0:
            verbosity = "INFO"
            import lmms_eval
            from lmms_eval import evaluator
            from lmms_eval.tasks import TaskManager, get_task_dict
            import os

            # Login to huggingface using token if not already logged in
            hf_token_path = ".hf_token"
            if not os.path.exists(hf_token_path):
                raise ValueError(f"HF token not found at {hf_token_path}")
            with open(hf_token_path, "r") as f:
                hf_token = f.read().strip()
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            task_manager = TaskManager(verbosity, model_name="prismatic")
            task_dict = get_task_dict(lmms_eval_list, task_manager)
            lm = lmms_eval.models.get_model("prismatic").create_from_arg_string(
                "",
                {
                    "batch_size": 1,
                    "max_batch_size": 1,
                    "device": "cuda:0",
                    "model_object": self.vlm,
                }
            )

            def _adjust_config(task_dict):
                adjusted_task_dict = {}
                for task_name, task_obj in task_dict.items():
                    if isinstance(task_obj, dict):
                        adjusted_task_dict = {
                            **adjusted_task_dict,
                            **{task_name: _adjust_config(task_obj)},
                        }

                    else:
                        task_obj = task_dict[task_name]
                        if type(task_obj) == tuple:
                            group, task_obj = task_obj
                            if task_obj is None:
                                continue
                        lm.task_dict[task_name] = task_obj.dataset


                        task_obj.set_config(key="num_fewshot", value=0)
                        # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                        task_obj.set_fewshot_seed(seed=1234)
                        # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                        adjusted_task_dict[task_name] = task_obj

                return adjusted_task_dict

            task_dict = _adjust_config(task_dict)

            results = evaluator.evaluate(
                lm=lm,
                task_dict=task_dict,
                limit=None,
                cache_requests=False,
                rewrite_requests_cache=False,
                bootstrap_iters=100000,
                write_out=False,
                log_samples=True,
                system_instruction=None,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
                verbosity=verbosity,
                cli_args=None,
            )
            print("results = ", results)

        # Other GPUs wait for completion. Hacky code since monitored_barrier is not implemented for NCCL backend and overwatch initializes the process group.
        dist.barrier()

          