# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from PIL import Image, ImageDraw

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)

        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins),
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        load_only_model_prefixes = self.checkpoint_conf.get("load_only_model_prefixes", None)
        if load_only_model_prefixes:
            load_only_model_prefixes = tuple(load_only_model_prefixes)
            original_key_count = len(model_state_dict)
            model_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith(load_only_model_prefixes)}
            logging.info(
                "Loading only model keys with prefixes %s: %d / %d keys kept.",
                load_only_model_prefixes,
                len(model_state_dict),
                original_key_count,
            )

        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
        if self.rank == 0:
            logging.info(
                f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}."
            )

        if load_only_model_prefixes and not self.checkpoint_conf.get("resume_training_state", False):
            logging.info("Skipping optimizer, scaler, and training-progress state after partial model load.")
            return

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            self.optims.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get("val", None), _recursive_=False)
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }

        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models=None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(
                self.seed_value + self.epoch * 100,
                self.max_epochs,
                self.distributed_rank,
            )

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)

            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "val"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = iters_per_epoch if self.limit_val_batches is None else self.limit_val_batches

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(batch, self.model, phase, loss_meters)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches

        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            # Step 1. 限制每个 epoch 的迭代数
            if data_iter >= limit_train_batches:
                break

            # measure data loading time
            # Step 2. 统计数据加载时间 上一个 batch 结束到当前 batch 真正开始处理之间花了多少时间数据准备/加载耗时
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            # Step 3. 对 batch 做预处理 关闭了 AMP 自动混合精度，用全精度执行
            # 之所以禁用 AMP，通常是因为：
            # 这些预处理操作更适合用 float32
            # 几何归一化这类计算对数值稳定性比较敏感
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            # Step 4. 把 batch 拷贝到设备上
            batch = copy_data_to_device(
                batch, self.device, non_blocking=True
            )  # non_blocking=True 如果条件满足，可以异步拷贝，提高吞吐

            # Step 5. 根据 accum_steps 把 batch 划分成若干个 chunk，每个 chunk 包含原 batch 的一部分数据。然后对每个 chunk 依次执行前向和反向传播，累积梯度。等所有 chunk 都处理完了，再执行一次优化器步骤来更新模型参数。
            # 这样做的好处是：
            # 显存不够时可以模拟更大的 batch size
            # 例如总 batch=16 放不下，可以拆成 4 次，每次算 4 个样本
            accum_steps = self.accum_steps

            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)
            # Step 6. 对每个 chunk 做前向和反向
            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            # compute gradient and do SGD step
            # Step 7. 计算当前训练进度 self.where，更新学习率调度器
            assert data_iter <= limit_train_batches  # allow for off by one errors
            # 比如 epoch=3，当前 batch 在本轮 50% 位置，那就是 3.5
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            # 比如总共 10 个 epoch，那么 3.5 / 10 = 0.35
            self.where = float(exact_epoch) / self.max_epochs  # where = 0.0 表示训练刚开始 where = 1.0 表示训练结束

            # Step 8. 根据当前训练进度更新 scheduler。
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )

            # Log schedulers
            # Step 9. 记录优化器和调度器参数到 TensorBoard
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else ("" + f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Step 10. 梯度裁剪和梯度监控 Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Step 11. 执行优化器更新 Optimizer step
            # 这一步结束后： 模型参数被更新 本次训练 iteration 才算真正完成
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Step 12. Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            mem.update(torch.cuda.max_memory_allocated() // 1e9)
            # 按设定频率输出训练日志，通常会显示：
            # 当前 epoch / iter
            # batch time
            # data time
            # 显存
            # loss
            # grad norm
            # 总耗时
            # 这就是终端里训练时看到的那类进度条/统计信息。
            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],  # 一个列表，里面每个元素是一个小 batch
        phase: str,  # 当前阶段，比如 "train" 或 "val"
        loss_meters: Dict[str, AverageMeter],  # 用来记录 loss、梯度等统计信息的计量器
    ):
        """
        把一个大 batch 拆成多个小 batch，每个小 batch 单独前向/反向，前几个 chunk 在 DDP 下不做梯度同步，最后一个 chunk 再同步，从而实现节省显存的梯度累积训练。
        """

        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """
        # Step 1. 先清空旧梯度,直接把梯度置为 None 可以让 PyTorch 在下一次反向传播时分配新的内存，减少内存碎片化，提高性能。
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        # 比如：
        # 原 batch size = 16
        # 拆成 4 个 chunk
        # 那么 accum_steps = 4
        accum_steps = len(chunked_batches)

        # Step 2. 设置自动混合精度 (AMP) 类型
        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        # Step 3. 对每个 chunk 做前向和反向传播，累积梯度
        for i, chunked_batch in enumerate(chunked_batches):
            # Step 4. 在 DDP 下决定要不要同步梯度
            # 所以这里的策略是：
            # 前 accum_steps - 1 个 chunk：用 no_sync()
            # 只做本地 backward
            # 不做分布式梯度同步
            # 最后一个 chunk：不用 no_sync()
            # 正常 backward
            # 这时才真正触发 DDP 梯度同步
            ddp_context = self.model.no_sync() if i < accum_steps - 1 else contextlib.nullcontext()

            with ddp_context:  # 外层 控制是否进行 DDP 梯度同步。
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):  # 内层 控制是否启用 AMP 以及使用哪种精度。
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"  # 构造日志名
                batch_size = chunked_batch["images"].shape[0]  # 当前这个 chunk 的样本数
                # 检查 loss 是否正常
                if not math.isfinite(loss.item()):  # nan  inf -inf 都不正常
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return
                # !为什么必须除？假设原始大 batch 被拆成 4 个 chunk，如果每个 chunk 都直接 backward 原始 loss，那么最终累积出来的总梯度会变成原来的 4 倍。
                loss /= accum_steps
                # 用 GradScaler 放大 loss 再反向传播    AMP 训练里的标准写法
                # 为什么要 scale(loss)？
                # 因为在 float16 下，小梯度可能会下溢成 0。
                # GradScaler 会先把 loss 放大，再做 backward，从而让梯度数值更稳定。
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images",
            "depths",
            "extrinsics",
            "intrinsics",
            "cam_points",
            "world_points",
            "point_masks",
        ]
        string_keys = ["seq_name"]

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, torch.flip(original_tensor, dims=[1])], dim=0)

        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2

        return batch

    def _cache_physical_metric_targets(self, batch: Mapping) -> Mapping:
        """Caches raw GT targets and normalization metadata for physical-unit logging."""
        if "extrinsics" in batch:
            batch["metric_raw_extrinsics"] = batch["extrinsics"].clone()

        if "depths" in batch:
            batch["metric_raw_depths"] = batch["depths"].clone()

        if all(key in batch for key in ["extrinsics", "world_points", "point_masks"]):
            ref_extrinsics = batch["extrinsics"][:, 0].clone()
            batch["metric_ref_extrinsics"] = ref_extrinsics

            world_points = batch["world_points"]
            point_masks = batch["point_masks"].float()
            rotation = ref_extrinsics[:, :3, :3]
            translation = ref_extrinsics[:, :3, 3]

            transformed_world_points = (
                world_points @ rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
            ) + translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            distances = transformed_world_points.norm(dim=-1)
            distance_sum = (distances * point_masks).sum(dim=[1, 2, 3])
            valid_count = point_masks.sum(dim=[1, 2, 3])
            avg_scale = (distance_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
            batch["metric_avg_scale"] = avg_scale

        return batch

    def _recover_physical_depths(
        self, pred_depth: torch.Tensor, data: Mapping
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Recovers predicted and GT depths in physical units for logging."""
        if not all(key in data for key in ["metric_avg_scale", "metric_raw_depths", "depth"]):
            return None

        scale = data["metric_avg_scale"].view(-1, 1, 1, 1)
        pred_depth_physical = pred_depth * scale
        gt_depth_physical = data["metric_raw_depths"].detach()
        return pred_depth_physical, gt_depth_physical

    def _recover_physical_extrinsics(self, pred_extrinsics: torch.Tensor, data: Mapping) -> Optional[torch.Tensor]:
        """Recovers predicted extrinsics from normalized coordinates back to raw scene coordinates."""
        if not all(
            key in data
            for key in [
                "metric_avg_scale",
                "metric_ref_extrinsics",
                "metric_raw_extrinsics",
            ]
        ):
            return None

        scale = data["metric_avg_scale"].view(-1, 1, 1)
        ref_extrinsics = data["metric_ref_extrinsics"].detach()

        pred_rotation_rel = pred_extrinsics[..., :3, :3]
        pred_translation_rel = pred_extrinsics[..., :3, 3] * scale

        ref_rotation = ref_extrinsics[:, None, :3, :3]
        ref_translation = ref_extrinsics[:, None, :3, 3]

        pred_rotation_abs = torch.matmul(pred_rotation_rel, ref_rotation)
        pred_translation_abs = (
            torch.matmul(pred_rotation_rel, ref_translation.unsqueeze(-1)).squeeze(-1) + pred_translation_rel
        )

        return torch.cat([pred_rotation_abs, pred_translation_abs.unsqueeze(-1)], dim=-1)

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        batch = self._cache_physical_metric_targets(batch)

        # Normalize camera extrinsics and points. The function returns new tensors.
        (
            normalized_extrinsics,
            normalized_cam_points,
            normalized_world_points,
            normalized_depths,
        ) = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.

        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass① 前向传播
        y_hat = model(images=batch["images"])

        # Loss computation ② 计算 loss
        loss_dict = self.loss(y_hat, batch)

        # Combine all data for logging ③ 记录标量和可视化
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _get_visual_frequency(self, phase: str) -> int:
        """Returns the TensorBoard visual logging frequency for the given phase."""
        visual_freq = getattr(self.logging_conf, "log_visual_frequency", None)
        if visual_freq is None:
            return 0

        if isinstance(visual_freq, Mapping):
            return int(visual_freq.get(phase, 0))

        if hasattr(visual_freq, phase):
            return int(getattr(visual_freq, phase))

        return 0

    def _robust_normalize_map(
        self,
        value_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        lower_q: float = 0.05,
        upper_q: float = 0.95,
    ) -> torch.Tensor:
        """Normalizes a 2D scalar map to [0, 1] using robust quantiles."""
        value_map = value_map.detach().float()
        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool()
            valid_values = value_map[valid_mask]
        else:
            valid_values = value_map.reshape(-1)

        valid_values = valid_values[torch.isfinite(valid_values)]
        if valid_values.numel() == 0:
            return torch.zeros_like(value_map)

        if valid_values.numel() == 1:
            normalized = torch.zeros_like(value_map)
        else:
            lo = torch.quantile(valid_values, lower_q)
            hi = torch.quantile(valid_values, upper_q)
            if not torch.isfinite(lo):
                lo = valid_values.min()
            if not torch.isfinite(hi):
                hi = valid_values.max()
            if (hi - lo).abs() < self.EPSILON:
                hi = lo + 1.0
            normalized = ((value_map - lo) / (hi - lo + self.EPSILON)).clamp(0.0, 1.0)

        if valid_mask is not None:
            normalized = torch.where(valid_mask, normalized, torch.zeros_like(normalized))
        return normalized

    def _scalar_map_to_rgb(
        self,
        value_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Converts a scalar map into a simple RGB heatmap for TensorBoard."""
        normalized = self._robust_normalize_map(value_map, valid_mask=valid_mask)
        red = normalized
        green = (1.0 - (2.0 * normalized - 1.0).abs()).clamp(0.0, 1.0)
        blue = 1.0 - normalized
        rgb = torch.stack([red, green, blue], dim=0)

        if valid_mask is not None:
            valid_mask = valid_mask.detach().bool().unsqueeze(0)
            rgb = torch.where(valid_mask, rgb, torch.zeros_like(rgb))
        return rgb.clamp(0.0, 1.0)

    def _compute_depth_error_statistics(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Computes summary statistics for valid depth differences."""
        valid_diff = (pred_depth - gt_depth)[valid_mask]
        valid_diff = valid_diff[torch.isfinite(valid_diff)]

        if valid_diff.numel() == 0:
            return None

        abs_diff = valid_diff.abs()
        return {
            "mean": valid_diff.mean().item(),
            "std": valid_diff.std(unbiased=False).item(),
            "min": valid_diff.min().item(),
            "max": valid_diff.max().item(),
            "mae": abs_diff.mean().item(),
        }

    def _append_text_panel_to_grid(
        self,
        image_grid: torch.Tensor,
        text_lines: Sequence[str],
    ) -> torch.Tensor:
        """Appends a text panel with summary lines below an image grid."""
        if len(text_lines) == 0:
            return image_grid

        image_grid = image_grid.detach().float().cpu().clamp(0.0, 1.0)
        height, width = image_grid.shape[1], image_grid.shape[2]

        line_height = 18
        top_bottom_padding = 12
        text_panel_height = max(48, top_bottom_padding * 2 + line_height * len(text_lines))

        panel_image = Image.new("RGB", (width, text_panel_height), color=(18, 18, 18))
        drawer = ImageDraw.Draw(panel_image)

        y_offset = top_bottom_padding
        for line in text_lines:
            drawer.text((12, y_offset), line, fill=(240, 240, 240))
            y_offset += line_height

        panel_array = np.asarray(panel_image).astype(np.float32) / 255.0
        panel_tensor = torch.from_numpy(panel_array).permute(2, 0, 1)

        return torch.cat([image_grid, panel_tensor], dim=1)

    def _log_derived_visual_scalars(self, data: Mapping, phase: str, step: int) -> None:
        """Logs additional scalar metrics derived from predictions for easier inspection."""
        if self.rank != 0 or step % self.logging_conf.log_freq != 0:
            return

        if "depth" in data and "depths" in data and "point_masks" in data:
            pred_depth = data["depth"].detach()[..., 0]
            recovered_depths = self._recover_physical_depths(pred_depth, data)
            if recovered_depths is not None:
                pred_depth, gt_depth = recovered_depths
            else:
                gt_depth = data["depths"].detach()
            valid_mask = data["point_masks"].detach().bool()
            valid_depth_error = (pred_depth - gt_depth).abs()[valid_mask]
            valid_depth_error = valid_depth_error[torch.isfinite(valid_depth_error)]
            if valid_depth_error.numel() > 0:
                self.tb_writer.log(
                    f"Metrics/{phase}/depth_abs_error_mean",
                    valid_depth_error.mean().item(),
                    step,
                )

        if "pose_enc" in data and "extrinsics" in data:
            pred_extrinsics, _ = pose_encoding_to_extri_intri(
                data["pose_enc"].detach(),
                image_size_hw=data["images"].shape[-2:],
            )
            recovered_extrinsics = self._recover_physical_extrinsics(pred_extrinsics, data)
            if recovered_extrinsics is not None:
                pred_extrinsics = recovered_extrinsics
                gt_extrinsics = data["metric_raw_extrinsics"].detach()
            else:
                gt_extrinsics = data["extrinsics"].detach()

            translation_error = torch.linalg.norm(
                pred_extrinsics[..., :3, 3] - gt_extrinsics[..., :3, 3],
                dim=-1,
            )
            translation_error = translation_error[torch.isfinite(translation_error)]
            if translation_error.numel() > 0:
                self.tb_writer.log(
                    f"Metrics/{phase}/pose_translation_error_mean",
                    translation_error.mean().item(),
                    step,
                )

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data["extrinsics"].shape[0]

        for key in keys_to_log:
            lookup_key = "objective" if key == "loss_objective" and "objective" in data else key
            if lookup_key in data:
                value = data[lookup_key].item() if torch.is_tensor(data[lookup_key]) else data[lookup_key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

        self._log_derived_visual_scalars(data, phase, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs TensorBoard visualizations for RGB and depth comparison."""
        visual_frequency = self._get_visual_frequency(phase)
        if not self.logging_conf.log_visuals or visual_frequency <= 0 or step % visual_frequency != 0:
            return

        if self.rank != 0 or "images" not in batch:
            return
        # 控制batch的索引
        # (B,S,C,H,W) 先选第 visual_batch_index 个样本，再选前 num_frames 帧来可视化
        visual_batch_index = min(
            int(getattr(self.logging_conf, "visual_batch_index", 0)),
            batch["images"].shape[0] - 1,
        )
        # 控制每个 batch 里要可视化多少帧，序列长度
        max_frames = max(1, int(getattr(self.logging_conf, "visual_max_frames", 4)))
        num_frames = min(batch["images"].shape[1], max_frames)

        depth_panels = []
        depth_stat_lines = []
        include_rgb = bool(getattr(self.logging_conf, "visual_include_rgb", True))
        include_depth = bool(getattr(self.logging_conf, "visual_include_depth", True))

        all_valid_depth_diffs = []

        for frame_idx in range(num_frames):
            frame_panels = []
            rgb = batch["images"][visual_batch_index, frame_idx].detach().float().cpu().clamp(0.0, 1.0)
            if include_rgb:
                frame_panels.append(rgb)

            if include_depth and all(key in batch for key in ["depth", "depths", "point_masks"]):
                # batch["depth"] 是模型预测的深度 (B,S,H,W,1)
                # batch["depths"] 是 GT 深度 (B,S,H,W,1)
                pred_depth = batch["depth"][visual_batch_index, frame_idx, ..., 0].detach().float().cpu()
                valid_mask = batch["point_masks"][visual_batch_index, frame_idx].detach().bool().cpu()

                recovered_depths = self._recover_physical_depths(batch["depth"].detach()[..., 0], batch)
                if recovered_depths is not None:
                    pred_depth_all, gt_depth_all = recovered_depths
                    pred_depth = pred_depth_all[visual_batch_index, frame_idx].detach().float().cpu()
                    gt_depth = gt_depth_all[visual_batch_index, frame_idx].detach().float().cpu()
                else:
                    gt_depth = batch["depths"][visual_batch_index, frame_idx].detach().float().cpu()

                depth_error = (pred_depth - gt_depth).abs()

                depth_stats = self._compute_depth_error_statistics(
                    pred_depth=pred_depth,
                    gt_depth=gt_depth,
                    valid_mask=valid_mask,
                )
                valid_diff = (pred_depth - gt_depth)[valid_mask]
                valid_diff = valid_diff[torch.isfinite(valid_diff)]
                if valid_diff.numel() > 0:
                    all_valid_depth_diffs.append(valid_diff)

                if depth_stats is not None:
                    depth_stat_lines.append(
                        "Frame {frame}: mean={mean:.4f}, std={std:.4f}, min={min:.4f}, max={max:.4f}, mae={mae:.4f}".format(
                            frame=frame_idx,
                            **depth_stats,
                        )
                    )
                else:
                    depth_stat_lines.append(f"Frame {frame_idx}: no valid depth pixels for statistics")

                frame_panels.extend(
                    [
                        self._scalar_map_to_rgb(gt_depth, valid_mask=valid_mask),
                        self._scalar_map_to_rgb(pred_depth, valid_mask=valid_mask),
                        self._scalar_map_to_rgb(depth_error, valid_mask=valid_mask),
                    ]
                )

            if frame_panels:
                depth_panels.append(torchvision.utils.make_grid(frame_panels, nrow=len(frame_panels), padding=4))

        if depth_panels:
            depth_grid = torchvision.utils.make_grid(depth_panels, nrow=1, padding=8).clamp(0.0, 1.0)

            if all_valid_depth_diffs:
                combined_valid_diff = torch.cat(all_valid_depth_diffs, dim=0)
                combined_stats = {
                    "mean": combined_valid_diff.mean().item(),
                    "std": combined_valid_diff.std(unbiased=False).item(),
                    "min": combined_valid_diff.min().item(),
                    "max": combined_valid_diff.max().item(),
                    "mae": combined_valid_diff.abs().mean().item(),
                }
                depth_stat_lines.insert(
                    0,
                    "Overall: mean={mean:.4f}, std={std:.4f}, min={min:.4f}, max={max:.4f}, mae={mae:.4f}".format(
                        **combined_stats
                    ),
                )

            depth_grid = self._append_text_panel_to_grid(depth_grid, depth_stat_lines)
            self.tb_writer.log_visuals(
                f"Visuals/{phase}/depth_comparison",
                depth_grid.numpy(),
                step,
            )


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {key: get_chunk_from_data(value, chunk_id, num_chunks) for key, value in data.items()}
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
