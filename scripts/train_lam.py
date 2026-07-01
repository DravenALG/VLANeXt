import argparse
import atexit
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.datasets.libero_lerobot_act import LiberoLeRobotAct, LiberoMixedLeRobotAct
from src.models.latent_action_models import build_lam_from_config
from src.utils.logging_utils import format_duration, progress_timing, setup_logger


SUPPORTED_LR_SCHEDULERS = {"cosine_decay": "cosine", "linear_decay": "linear", "fixed": "constant_with_warmup"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def to_tensor_image(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 127.5 - 1.0


class DataCollatorForLAM:
    def __init__(self, view_mode="single", augmentation=None):
        self.view_mode = view_mode
        self.augmentation = augmentation or {}
        self.aug_enabled = bool(self.augmentation.get("enabled", True))
        self.brightness = float(self.augmentation.get("brightness", 0.1))
        self.contrast = float(self.augmentation.get("contrast", 0.1))
        self.noise_std = float(self.augmentation.get("noise_std", 0.01))

    def __call__(self, batch):
        frames_t = []
        frames_tp1 = []
        frames_t_aug = []
        frames_tp1_aug = []
        for sample in batch:
            frame_t = self._sample_frame(sample, future=False)
            frame_tp1 = self._sample_frame(sample, future=True)
            frames_t.append(frame_t)
            frames_tp1.append(frame_tp1)
            aug_t, aug_tp1 = self._augment_pair(frame_t, frame_tp1)
            frames_t_aug.append(aug_t)
            frames_tp1_aug.append(aug_tp1)
        return {
            "frame_t": torch.stack(frames_t),
            "frame_tp1": torch.stack(frames_tp1),
            "frame_t_aug": torch.stack(frames_t_aug),
            "frame_tp1_aug": torch.stack(frames_tp1_aug),
        }

    def _sample_frame(self, sample, future=False):
        if future:
            main = to_tensor_image(sample["future_image"])
            if self.view_mode == "multi":
                wrist = to_tensor_image(sample["future_image_wrist"])
                return torch.cat([main, wrist], dim=0)
            return main
        main = to_tensor_image(sample["image"])
        if self.view_mode == "multi":
            wrist = to_tensor_image(sample["image_wrist"])
            return torch.cat([main, wrist], dim=0)
        return main

    def _augment_pair(self, frame_t, frame_tp1):
        if not self.aug_enabled:
            return frame_t.clone(), frame_tp1.clone()
        brightness = float(torch.empty(1).uniform_(-self.brightness, self.brightness))
        contrast = float(torch.empty(1).uniform_(1.0 - self.contrast, 1.0 + self.contrast))

        def aug(x):
            y = x * contrast + brightness
            if self.noise_std > 0:
                y = y + torch.randn_like(y) * self.noise_std
            return y.clamp(-1.0, 1.0)

        return aug(frame_t), aug(frame_tp1)


def scheduler_name(config):
    name = str(config.get("train", {}).get("scheduler", "cosine_decay")).lower()
    if name not in SUPPORTED_LR_SCHEDULERS:
        raise ValueError(f"Unknown train.scheduler={name}; options={sorted(SUPPORTED_LR_SCHEDULERS)}")
    return name


def build_scheduler(config, optimizer):
    name = scheduler_name(config)
    kwargs = {
        "name": SUPPORTED_LR_SCHEDULERS[name],
        "optimizer": optimizer,
        "num_warmup_steps": int(config["train"].get("warmup_steps", 0)),
    }
    if name != "fixed":
        kwargs["num_training_steps"] = int(config["data"]["max_steps"])
    return get_scheduler(**kwargs)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if not first_key.startswith("module."):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_checkpoint(path, model, optimizer, lr_scheduler, step, config):
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def build_save_dir(config):
    project_name = config["project"]["name"]
    output_dir = config["project"].get("output_dir", "./checkpoints")
    return os.path.join(output_dir, project_name)


def build_dataloader(config, per_device_batch_size, seed):
    data_cfg = config["data"]
    task_suite = data_cfg.get("task_suite_name", "libero_mixed")
    data_root = data_cfg["data_root"]
    view_mode = data_cfg.get("view_mode", "multi")
    future_len = int(data_cfg.get("future_len", 1))
    if future_len != 1:
        raise ValueError("LAM training expects data.future_len=1 for adjacent t -> t+1 pairs.")

    common_kwargs = dict(
        history_len=1,
        future_len=1,
        full_sequence=bool(data_cfg.get("full_sequence", True)),
        input_modality="image",
        view_mode=view_mode,
        load_future_image=True,
        future_image_mode="horizon",
        buffer_size=data_cfg.get("buffer_size", 10000),
        sampling_rate=data_cfg.get("sampling_rate", 1.0),
        allow_end_padding=False,
        emit_proprioception=False,
        emit_history_actions=False,
        batch_size=per_device_batch_size,
        drop_last=bool(data_cfg.get("drop_last", True)),
        seed=seed,
        image_resize_size=data_cfg.get("image_resize_size", 128),
        action_mode=data_cfg.get("action_mode", "libero"),
        length=data_cfg.get("length", None),
    )
    if task_suite == "libero_mixed":
        dataset = LiberoMixedLeRobotAct(
            data_root=data_root,
            normalization_suite_name=data_cfg.get("normalization_suite_name", "libero_mixed"),
            **common_kwargs,
        )
    else:
        dataset = LiberoLeRobotAct(
            data_path=os.path.join(data_root, task_suite),
            dataset_name=task_suite,
            normalization_suite_name=data_cfg.get("normalization_suite_name", task_suite),
            **common_kwargs,
        )

    collator = DataCollatorForLAM(view_mode=view_mode, augmentation=data_cfg.get("augmentation", {}))
    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": None,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **loader_kwargs)


def setup_distributed(config):
    distributed = bool(config["train"].get("distributed", False))
    if distributed:
        dist.init_process_group(backend=config["train"].get("dist_backend", "nccl"))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    return distributed, local_rank, global_rank, world_size, device


def train(config):
    atexit.register(cleanup)
    seed = int(config["train"].get("seed", 42))
    set_seed(seed)
    is_distributed, local_rank, global_rank, world_size, device = setup_distributed(config)
    grad_accum = int(config["train"].get("gradient_accumulation_steps", 1))
    total_batch_size = int(config["data"]["batch_size"])
    per_device_batch_size = total_batch_size // (world_size * grad_accum)
    if per_device_batch_size <= 0:
        raise ValueError("data.batch_size must be >= world_size * gradient_accumulation_steps")

    save_dir = build_save_dir(config)
    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_config(config, os.path.join(save_dir, "config.yaml"))
    logger = setup_logger(
        "train_lam",
        os.path.join(save_dir, "train_lam.log") if global_rank == 0 else None,
        rank=global_rank,
    )
    if global_rank == 0:
        logger.info(
            "Starting LAM training save_dir=%s distributed=%s world_size=%d per_device_batch_size=%d",
            save_dir,
            is_distributed,
            world_size,
            per_device_batch_size,
        )

    dataloader = build_dataloader(config, per_device_batch_size, seed)
    model = build_lam_from_config(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"].get("learning_rate", 3.0e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1.0e-4)),
    )
    lr_scheduler = build_scheduler(config, optimizer)

    start_step = 0
    resume_path = config["train"].get("resume_path", "")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        if global_rank == 0:
            logger.info("Resumed LAM checkpoint %s at step %d", resume_path, start_step)

    use_wandb = bool(config["project"].get("use_wandb", False)) and global_rank == 0
    if use_wandb:
        import wandb

        wandb.init(
            project=config["project"].get("wandb_project", "VLANeXt-LAM"),
            entity=config["project"].get("wandb_entity", None),
            name=config["project"]["name"],
            config=config,
        )
    else:
        wandb = None

    max_steps = int(config["data"]["max_steps"])
    log_interval = int(config["project"].get("log_interval", 10))
    save_interval = int(config["project"].get("save_interval", 1000))
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    autocast_dtype = torch.bfloat16 if config["train"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and bool(config["train"].get("autocast", True))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(dataloader)
    step = start_step
    batch_idx = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = step

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            outputs = model(**batch)
            loss = outputs["loss"] / grad_accum
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            should_log = global_rank == 0 and (step % log_interval == 0 or step == max_steps)
            if should_log:
                now = time.time()
                completed = step - start_step
                total = max_steps - start_step
                elapsed, avg_step_time, eta = progress_timing(start_time, completed, total, now=now)
                recent_steps = max(1, step - last_log_step)
                recent_step_time = (now - last_log_time) / recent_steps
                last_log_time = now
                last_log_step = step
                loss_value = float(outputs["loss"].detach().cpu())
                recon_value = float(outputs["recon_loss"].detach().cpu())
                lr_value = lr_scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d/%d loss=%.6f recon=%.6f lr=%.6e elapsed=%s avg_step=%.2fs recent_step=%.2fs eta=%s",
                    step,
                    max_steps,
                    loss_value,
                    recon_value,
                    lr_value,
                    format_duration(elapsed),
                    avg_step_time,
                    recent_step_time,
                    format_duration(eta),
                )
                if wandb is not None:
                    log = {"train/loss": loss_value, "train/recon_loss": recon_value, "train/lr": lr_value, "step": step}
                    for key, value in outputs.items():
                        if key.endswith("_loss") and key != "recon_loss":
                            log[f"train/{key}"] = float(value.detach().cpu())
                    wandb.log(log)

            if global_rank == 0 and step % save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, step, config)
                logger.info("Saved checkpoint to %s", ckpt_path)

        batch_idx += 1

    if global_rank == 0:
        final_path = os.path.join(save_dir, "checkpoint_final.pt")
        save_checkpoint(final_path, model, optimizer, lr_scheduler, step, config)
        logger.info("Saved final checkpoint to %s", final_path)
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_train_lam_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train(load_config(args.config))
