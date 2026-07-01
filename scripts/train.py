import os
import yaml
import argparse
import atexit
import wandb
import random
import sys
import time
from PIL import Image

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from contextlib import nullcontext
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TVF

from src.models.VLANeXt import VLANeXt
from src.datasets.libero_act import LiberoAct, LiberoMixedAct, get_libero_normalization_suite_name
from src.datasets.libero_lerobot_act import LiberoLeRobotAct, LiberoMixedLeRobotAct
from src.datasets.droid_lerobot_act import DroidLeRobotAct
from src.datasets.real_world_act import RealWorldAct
from src.datasets.language_action import (
    format_language_action_prompt,
    normalize_language_action_format,
)
from src.evaluation.speed_size_utils import run_size_speed_eval
from src.utils.logging_utils import format_duration, progress_timing, setup_logger


# -----------------------------------------------------------------------------
# ------------------------------ Preliminary ----------------------------------
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _cleanup_on_exit():
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# ----------------------------- Data Collector --------------------------------
# -----------------------------------------------------------------------------
DEFAULT_WAN_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the instruction: {instruction}"


class DataCollatorForVLANeXt:
    def __init__(
        self,
        processor,
        use_proprio_input_vlm=True,
        use_action_input_policy=True,
        input_modality="video",
        view_mode="single",
        fps=20.0,
        augmentation=None,
        bimanual=False,
        load_future_image=False,
        future_image_prediction_type="emu_token",
        future_image_processor=None,
        load_language_action=False,
        language_action_format="lap",
        language_action_num_bins=1000,
        load_future_video=False,
        wan_prompt_template=DEFAULT_WAN_PROMPT_TEMPLATE,
        future_len=1,
        action_dim=7,
    ):
        self.processor = processor
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.fps = float(fps)
        self.load_future_image = load_future_image
        self.future_image_prediction_type = future_image_prediction_type
        self.future_image_processor = future_image_processor
        self.load_language_action = bool(load_language_action)
        self.language_action_format = normalize_language_action_format(language_action_format)
        self.language_action_num_bins = int(language_action_num_bins)
        self.load_future_video = bool(load_future_video)
        self.wan_prompt_template = str(wan_prompt_template or DEFAULT_WAN_PROMPT_TEMPLATE)
        self.future_len = int(future_len)
        self.action_dim = int(action_dim)

        self.bimanual = bimanual

        self.aug = augmentation or {}
        self.aug_enabled = bool(self.aug.get("enabled", False))

        rrc = self.aug.get("random_resized_crop", {}) or {}
        self.rrc_scale = tuple(rrc.get("scale", (0.9, 0.9)))
        self.rrc_ratio = tuple(rrc.get("ratio", (1.0, 1.0)))

        self.rb = self.aug.get("random_brightness", None)
        self.rc = self.aug.get("random_contrast", None)
        self.rs = self.aug.get("random_saturation", None)
        self.rh = self.aug.get("random_hue", None)
        self.augment_order = list(self.aug.get("augment_order", []))

    # image augmentations
    def _augment_frames_uint8(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentation to frames (T,H,W,C) or (H,W,C)."""
        if (not self.aug_enabled) or (not self.augment_order):
            return frames

        is_video = (frames.ndim == 4)
        pil_frames = [self._to_pil(f) for f in (frames if is_video else [frames])]
        out_h, out_w = pil_frames[0].height, pil_frames[0].width

        crop_params = None
        if "random_resized_crop" in self.augment_order and self.aug.get("random_resized_crop", None) is not None:
            i, j, h, w = RandomResizedCrop.get_params(pil_frames[0], scale=self.rrc_scale, ratio=self.rrc_ratio)
            crop_params = (i, j, h, w)

        b_fac = self._sample_brightness_factor() if "random_brightness" in self.augment_order else 1.0
        c_fac = self._sample_contrast_factor() if "random_contrast" in self.augment_order else 1.0
        s_fac = self._sample_saturation_factor() if "random_saturation" in self.augment_order else 1.0
        h_del = self._sample_hue_delta() if "random_hue" in self.augment_order else 0.0
        h_del = float(np.clip(h_del, -0.5, 0.5))

        out = []
        for pil in pil_frames:
            for op in self.augment_order:
                if op == "random_resized_crop" and crop_params is not None:
                    i, j, h, w = crop_params
                    pil = TVF.resized_crop(pil, i, j, h, w, size=(out_h, out_w))
                elif op == "random_brightness":
                    pil = TVF.adjust_brightness(pil, b_fac)
                elif op == "random_contrast":
                    pil = TVF.adjust_contrast(pil, c_fac)
                elif op == "random_saturation":
                    pil = TVF.adjust_saturation(pil, s_fac)
                elif op == "random_hue":
                    pil = TVF.adjust_hue(pil, h_del)
                else:
                    pass
            out.append(np.asarray(pil, dtype=np.uint8))

        out = np.stack(out, axis=0)
        return out if is_video else out[0]
    
    def _sample_brightness_factor(self) -> float:
        if not self.rb:
            return 1.0
        if len(self.rb) == 1:
            x = float(self.rb[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rb[0]), float(self.rb[1]))

    def _sample_contrast_factor(self) -> float:
        if not self.rc:
            return 1.0
        if len(self.rc) == 1:
            x = float(self.rc[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rc[0]), float(self.rc[1]))

    def _sample_saturation_factor(self) -> float:
        if not self.rs:
            return 1.0
        if len(self.rs) == 1:
            x = float(self.rs[0])
            return self._uniform(1.0 - x, 1.0 + x)
        return self._uniform(float(self.rs[0]), float(self.rs[1]))

    def _sample_hue_delta(self) -> float:
        if not self.rh:
            return 0.0
        if len(self.rh) == 1:
            x = float(self.rh[0])
            return self._uniform(-x, x)
        return self._uniform(float(self.rh[0]), float(self.rh[1]))

    def _to_pil(self, img_np: np.ndarray) -> Image.Image:
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _uniform(self, a: float, b: float) -> float:
        return float(np.random.uniform(a, b))

    # utilities for future image prediction
    def _future_image_to_tensor(self, image) -> torch.Tensor:
        image_np = np.asarray(image)
        if image_np.dtype != np.uint8:
            if np.issubdtype(image_np.dtype, np.floating):
                image_np = np.clip(image_np, 0.0, 1.0) * 255.0
            image_np = image_np.astype(np.uint8)
        return torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 127.5 - 1.0

    def _future_images_to_dino_tensor(self, images) -> torch.Tensor:
        if self.future_image_processor is None:
            raise ValueError("future_image_processor is required for dinov3_flow future image prediction.")
        processed = self.future_image_processor(images=images, return_tensors="pt")
        return processed["pixel_values"]

    def _future_video_to_tensor(self, video) -> torch.Tensor:
        video_np = np.asarray(video)
        if video_np.ndim != 4:
            raise ValueError(f"future_video must be [T, H, W, C], got {tuple(video_np.shape)}.")
        if video_np.dtype != np.uint8:
            if np.issubdtype(video_np.dtype, np.floating):
                video_np = np.clip(video_np, 0.0, 1.0) * 255.0
            video_np = video_np.astype(np.uint8)
        return torch.from_numpy(video_np.copy()).permute(3, 0, 1, 2).float() / 127.5 - 1.0

    # utilities for language action learning
    # generate the text prompt for language action learning
    def _format_language_action_text_prompts(self, prefix: str, answer: str) -> str:
        text = prefix + answer
        eos_text = self._get_eos_text()
        if eos_text and not text.endswith(eos_text):
            text += eos_text
        return text
    
    def _get_eos_text(self) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
        return eos_token or ""

    # generate language action labels
    @staticmethod
    def _add_language_action_labels(full_inputs, prefix_inputs):
        language_action_labels = full_inputs["input_ids"].clone()
        full_attention = full_inputs.get("attention_mask")
        if full_attention is not None:
            language_action_labels = language_action_labels.masked_fill(full_attention == 0, -100)

        prefix_attention = prefix_inputs.get("attention_mask")
        if prefix_attention is None:
            prefix_lengths = [prefix_inputs["input_ids"].shape[1]] * language_action_labels.shape[0]
        else:
            prefix_lengths = prefix_attention.long().sum(dim=1).tolist()

        for i, prefix_len in enumerate(prefix_lengths):
            prefix_len = int(prefix_len)
            if full_attention is None:
                language_action_labels[i, : min(prefix_len, language_action_labels.shape[1])] = -100
                continue
            nonpad = torch.nonzero(full_attention[i].bool(), as_tuple=False).flatten()
            if nonpad.numel() > 0:
                language_action_labels[i, nonpad[: min(prefix_len, nonpad.numel())]] = -100

        full_inputs["language_action_labels"] = language_action_labels
        return full_inputs

    def __call__(self, batch):
        texts = []
        language_action_prefix_texts = []
        language_action_full_texts = []
        fps = self.fps

        images = []
        videos = []
        
        gt_actions_list = []
        proprio_list = []
        hist_actions_list = []
        future_images_list = []
        future_videos_list = []

        processor_name = self.processor.__class__.__name__
        is_wan = "Wan" in processor_name
        is_paligemma = "PaliGemma" in processor_name
        is_qwen = "Qwen" in processor_name
        is_llama = "Llama" in processor_name
        if is_wan and self.load_language_action:
            raise ValueError("language-action loss is disabled for WAN video-generation backbone.")

        for sample in batch:
            instruction = sample["instruction"]
            language_action_answer = None
            language_action_prompt = None
            if self.load_language_action:
                language_action_answer = str(sample["language_action_text"])
                frame_description = (
                    "normalized action space"
                    if self.language_action_format == "vla-0"
                    else "robot base frame"
                )
                language_action_prompt = (
                    format_language_action_prompt(
                        instruction,
                        language_action_format=self.language_action_format,
                        frame_description=frame_description,
                        future_len=self.future_len,
                        action_dim=self.action_dim,
                    )
                    + "\nAnswer: "
                )

            if is_wan:
                texts.append(self.wan_prompt_template.format(instruction=instruction, task=instruction))

            elif is_paligemma:
                im0 = self._augment_frames_uint8(sample["image"])
                num_imgs = 1
                if self.view_mode == "multi":
                    im1 = self._augment_frames_uint8(sample["image_wrist"])
                    if self.bimanual and "image_wrist2" in sample:
                        im2 = self._augment_frames_uint8(sample["image_wrist2"])
                        images.extend([im0, im1, im2])
                        num_imgs = 3
                    else:
                        images.extend([im0, im1])
                        num_imgs = 2
                else:
                    images.append(im0)

                texts.append("<image>" * num_imgs + instruction)
                if self.load_language_action:
                    la_prefix = "<image>" * num_imgs + language_action_prompt
                    language_action_prefix_texts.append(la_prefix)
                    language_action_full_texts.append(
                        self._format_language_action_text_prompts(la_prefix, language_action_answer)
                    )

            elif is_llama:
                im0 = self._augment_frames_uint8(sample["image"])
                if self.view_mode == "multi":
                    im1 = self._augment_frames_uint8(sample["image_wrist"])
                    if self.bimanual and "image_wrist2" in sample:
                        im2 = self._augment_frames_uint8(sample["image_wrist2"])
                        images.extend([im0, im1, im2])
                    else:
                        images.extend([im0, im1])
                else:
                    images.append(im0)

                texts.append(instruction)
                if self.load_language_action:
                    language_action_prefix_texts.append(language_action_prompt)
                    language_action_full_texts.append(
                        self._format_language_action_text_prompts(language_action_prompt, language_action_answer)
                    )

            elif is_qwen:
                content = []
                if self.input_modality == "video":
                    v0 = self._augment_frames_uint8(sample["video"])
                    if self.view_mode == "multi":
                        v1 = self._augment_frames_uint8(sample["video_wrist"])
                        if self.bimanual and "video_wrist2" in sample:
                            v2 = self._augment_frames_uint8(sample["video_wrist2"])
                            content.extend([
                                {"type": "video", "video": v0},
                                {"type": "video", "video": v1},
                                {"type": "video", "video": v2},
                            ])
                            videos.extend([v0, v1, v2])
                        else:
                            content.extend([{"type": "video", "video": v0}, {"type": "video", "video": v1}])
                            videos.extend([v0, v1])
                    else:
                        content.append({"type": "video", "video": v0})
                        videos.append(v0)

                elif self.input_modality == "image":
                    im0 = self._augment_frames_uint8(sample["image"])
                    if self.view_mode == "multi":
                        im1 = self._augment_frames_uint8(sample["image_wrist"])
                        if self.bimanual and "image_wrist2" in sample:
                            im2 = self._augment_frames_uint8(sample["image_wrist2"])
                            content.extend([
                                {"type": "image", "image": im0},
                                {"type": "image", "image": im1},
                                {"type": "image", "image": im2},
                            ])
                            images.extend([im0, im1, im2])
                        else:
                            content.extend([{"type": "image", "image": im0}, {"type": "image", "image": im1}])
                            images.extend([im0, im1])
                    else:
                        content.append({"type": "image", "image": im0})
                        images.append(im0)
                else:
                    raise ValueError(f"Unknown input_modality: {self.input_modality}")

                if self.load_language_action:
                    la_content = list(content)
                    la_content.append({"type": "text", "text": language_action_prompt})
                    la_messages = [{"role": "user", "content": la_content}]
                    la_prefix = self.processor.apply_chat_template(
                        la_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    language_action_prefix_texts.append(la_prefix)
                    language_action_full_texts.append(
                        self._format_language_action_text_prompts(la_prefix, language_action_answer)
                    )

                content.append({"type": "text", "text": instruction})

                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)

            gt_actions_list.append(sample["future_actions"])
            if self.use_proprio_input_vlm:
                proprio_list.append(sample["proprioception"])
            if self.use_action_input_policy:
                hist_actions_list.append(sample["history_actions"])
            if self.load_future_image:
                if "future_image" not in sample:
                    raise KeyError("load_future_image=True but sample has no `future_image` field.")
                if self.future_image_prediction_type == "dinov3_flow":
                    future_images_list.append(sample["future_image"])
                else:
                    future_images_list.append(self._future_image_to_tensor(sample["future_image"]))
            if self.load_future_video:
                if "future_video" not in sample:
                    raise KeyError("load_future_video=True but sample has no `future_video` field.")
                future_video = self._augment_frames_uint8(sample["future_video"])
                future_videos_list.append(self._future_video_to_tensor(future_video))

        model_texts = language_action_full_texts if self.load_language_action else texts

        if is_wan:
            inputs = {"prompt_texts": model_texts}
        elif is_paligemma:
            inputs = self.processor(
                text=model_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_llama:
            inputs = self.processor.tokenizer(
                model_texts,
                padding=True,
                return_tensors="pt"
            )
            image_inputs = self.processor.image_processor(
                images,
                return_tensors="pt"
            )
            inputs["pixel_values"] = image_inputs["pixel_values"]
        elif is_qwen:
            if self.input_modality == "video":
                video_metadata = [
                    {"total_num_frames": v.shape[0], "fps": fps, "frames_indices": list(range(v.shape[0]))}
                    for v in videos
                ]
                inputs = self.processor(
                    text=model_texts,
                    videos=videos,
                    videos_kwargs={"fps": fps, "return_metadata": True, "video_metadata": video_metadata}, 
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=model_texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )

        if self.load_language_action:
            if is_paligemma:
                language_action_prefix_inputs = self.processor(
                    text=language_action_prefix_texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
            elif is_llama:
                language_action_prefix_inputs = self.processor.tokenizer(
                    language_action_prefix_texts,
                    padding=True,
                    return_tensors="pt",
                )
                image_inputs = self.processor.image_processor(
                    images,
                    return_tensors="pt",
                )
                language_action_prefix_inputs["pixel_values"] = image_inputs["pixel_values"]
            elif is_qwen:
                if self.input_modality == "video":
                    video_metadata = [
                        {"total_num_frames": v.shape[0], "fps": fps, "frames_indices": list(range(v.shape[0]))}
                        for v in videos
                    ]
                    language_action_prefix_inputs = self.processor(
                        text=language_action_prefix_texts,
                        videos=videos,
                        videos_kwargs={
                            "fps": fps,
                            "return_metadata": True,
                            "video_metadata": video_metadata,
                        },
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    language_action_prefix_inputs = self.processor(
                        text=language_action_prefix_texts,
                        images=images,
                        padding=True,
                        return_tensors="pt",
                    )
            inputs = self._add_language_action_labels(
                inputs,
                language_action_prefix_inputs,
            )

        gt_actions = torch.stack(gt_actions_list)
        proprio = torch.stack(proprio_list) if self.use_proprio_input_vlm else None
        hist_actions = torch.stack(hist_actions_list) if self.use_action_input_policy else None
        future_images = None
        if self.load_future_image:
            if self.future_image_prediction_type == "dinov3_flow":
                future_images = self._future_images_to_dino_tensor(future_images_list)
            else:
                future_images = torch.stack(future_images_list)
        future_videos = torch.stack(future_videos_list) if self.load_future_video else None

        return inputs, gt_actions, proprio, hist_actions, future_images, future_videos

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# ---------------------------- Scheduler Helper -------------------------------
# -----------------------------------------------------------------------------
SUPPORTED_LR_SCHEDULERS = ("cosine_decay", "linear_decay", "fixed")


def get_lr_scheduler_name(config):
    scheduler_name = config.get('train', {}).get('scheduler', 'cosine_decay')
    if scheduler_name is None:
        scheduler_name = 'cosine_decay'
    scheduler_name = str(scheduler_name).strip().lower()
    if scheduler_name not in SUPPORTED_LR_SCHEDULERS:
        raise ValueError(
            f"Unknown train.scheduler: {scheduler_name}. "
            f"Options are: {', '.join(SUPPORTED_LR_SCHEDULERS)}."
        )
    return scheduler_name


def build_torch_lr_scheduler(scheduler_name, optimizer, num_warmup_steps, num_training_steps):
    scheduler_type = {
        "cosine_decay": "cosine",
        "linear_decay": "linear",
        "fixed": "constant_with_warmup",
    }[scheduler_name]

    kwargs = {
        "optimizer": optimizer,
        "num_warmup_steps": int(num_warmup_steps),
    }
    if scheduler_name != "fixed":
        kwargs["num_training_steps"] = int(num_training_steps)

    return get_scheduler(scheduler_type, **kwargs)


def build_deepspeed_scheduler_config(scheduler_name, learning_rate, warmup_steps, total_steps):
    learning_rate = float(learning_rate)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    if scheduler_name == "cosine_decay":
        return {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": 0.0,
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps,
                "warmup_type": "linear",
            }
        }

    if scheduler_name == "linear_decay":
        return {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps,
                "warmup_type": "linear",
            }
        }

    if scheduler_name == "fixed":
        return {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
                "warmup_type": "linear",
            }
        }

    raise ValueError(
        f"Unknown train.scheduler: {scheduler_name}. "
        f"Options are: {', '.join(SUPPORTED_LR_SCHEDULERS)}."
    )

# -----------------------------------------------------------------------------
# --------------------------- Deepspeed helper --------------------------------
# -----------------------------------------------------------------------------
def build_deepspeed_config(config, per_device_batch_size, learning_rate=None, warmup_steps=None, total_steps=None):
    """Build a DeepSpeed config dict from the YAML config."""
    train_cfg = config['train']
    ds_cfg = train_cfg.get('deepspeed', {})
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    learning_rate = float(train_cfg['learning_rate'] if learning_rate is None else learning_rate)
    warmup_steps = int(train_cfg['warmup_steps'] if warmup_steps is None else warmup_steps)
    total_steps = int(config['data']['max_steps'] if total_steps is None else total_steps)
    scheduler_name = get_lr_scheduler_name(config)

    ds_config = {
        "train_micro_batch_size_per_gpu": int(per_device_batch_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": train_cfg.get('max_grad_norm', 1.0),
        "steps_per_print": config['project'].get('log_interval', 10),
        "zero_allow_untested_optimizer": True,
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": ds_cfg.get('zero_stage', 2),
            "offload_optimizer": {
                "device": ds_cfg.get('offload_optimizer_device', 'none'),
            },
            "offload_param": {
                "device": ds_cfg.get('offload_param_device', 'none'),
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": int(float(ds_cfg.get('reduce_bucket_size', 5e8))),
            "allgather_bucket_size": int(float(ds_cfg.get('allgather_bucket_size', 5e8))),
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "weight_decay": float(train_cfg['weight_decay']),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            }
        },
        "scheduler": build_deepspeed_scheduler_config(
            scheduler_name,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
        "activation_checkpointing": {
            "partition_activations": ds_cfg.get('partition_activations', False),
            "cpu_checkpointing": ds_cfg.get('cpu_checkpointing', False),
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
        },
        "wall_clock_breakdown": False,
    }
    return ds_config


def _torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _state_dict_has_module_prefix(state_dict):
    first_key = next(iter(state_dict), "")
    return first_key.startswith("module.")


def _strip_module_prefix(state_dict):
    if not _state_dict_has_module_prefix(state_dict):
        return state_dict
    prefix_len = len("module.")
    return {k[prefix_len:]: v for k, v in state_dict.items()}


def _add_module_prefix(state_dict):
    if _state_dict_has_module_prefix(state_dict):
        return state_dict
    return {f"module.{k}": v for k, v in state_dict.items()}


def _find_deepspeed_model_state_file(tag_dir):
    preferred = os.path.join(tag_dir, "mp_rank_00_model_states.pt")
    if os.path.exists(preferred):
        return preferred

    if not os.path.isdir(tag_dir):
        return None
    candidates = sorted(
        os.path.join(tag_dir, name)
        for name in os.listdir(tag_dir)
        if name.endswith("_model_states.pt")
    )
    return candidates[0] if candidates else None


def _resolve_deepspeed_checkpoint(load_dir):
    if not os.path.isdir(load_dir):
        raise ValueError(f"DeepSpeed checkpoint load_dir must be a directory: {load_dir}")

    load_dir = os.path.normpath(load_dir)
    latest_path = os.path.join(load_dir, "latest")
    if not os.path.isfile(latest_path):
        raise FileNotFoundError(f"DeepSpeed checkpoint load_dir has no `latest` file: {load_dir}")

    with open(latest_path, "r") as f:
        tag = f.read().strip()

    if not tag:
        raise ValueError(f"DeepSpeed checkpoint latest file is empty: {latest_path}")
    tag_dir = os.path.join(load_dir, tag)
    if not os.path.isdir(tag_dir):
        raise FileNotFoundError(f"DeepSpeed checkpoint tag dir does not exist: {tag_dir}")

    model_state_file = _find_deepspeed_model_state_file(tag_dir)
    if model_state_file is None:
        raise FileNotFoundError(
            f"DeepSpeed checkpoint tag dir has no `*_model_states.pt` file: {tag_dir}"
        )
    return load_dir, tag, tag_dir, model_state_file


def _load_deepspeed_client_state(tag_dir, map_location="cpu"):
    model_state_file = _find_deepspeed_model_state_file(tag_dir)
    if model_state_file is None:
        return {}
    checkpoint = _torch_load_checkpoint(model_state_file, map_location=map_location)
    return {
        k: v
        for k, v in checkpoint.items()
        if k not in {
            "module",
            "optimizer",
            "lr_scheduler",
            "buffer_names",
            "param_shapes",
            "frozen_param_shapes",
            "shared_params",
            "frozen_param_fragments",
        }
    }


def _load_model_state_dict_from_checkpoint(checkpoint_path, map_location="cpu"):
    if os.path.isfile(checkpoint_path):
        checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=map_location)
        return checkpoint["model_state_dict"], {"type": "file", "step": checkpoint.get("step", 0)}

    load_dir, tag, tag_dir, _ = _resolve_deepspeed_checkpoint(checkpoint_path)
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    state_dict = get_fp32_state_dict_from_zero_checkpoint(load_dir, tag=tag)
    client_state = _load_deepspeed_client_state(tag_dir)
    return state_dict, {
        "type": "deepspeed",
        "load_dir": load_dir,
        "tag": tag,
        "step": client_state.get("step", 0),
    }


def _drop_mismatched_state_dict_entries(model, state_dict, logger=None):
    reference = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        ref_value = reference.get(key)
        if ref_value is not None and tuple(ref_value.shape) != tuple(value.shape):
            skipped.append((key, tuple(value.shape), tuple(ref_value.shape)))
            continue
        filtered[key] = value

    if skipped and logger is not None:
        preview = ", ".join(
            f"{key}: ckpt{src}->model{dst}"
            for key, src, dst in skipped[:10]
        )
        if len(skipped) > 10:
            preview += f", ... (+{len(skipped) - 10} more)"
        logger.warning("Skipping %d mismatched pretrained tensors: %s", len(skipped), preview)
    return filtered, skipped

# -----------------------------------------------------------------------------
# ----------------------------- Ema Helper ------------------------------------
# -----------------------------------------------------------------------------
def _clone_tensor_for_ema(tensor):
    tensor = tensor.detach().cpu().clone()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


def _clone_state_dict_for_ema(state_dict):
    return {k: _clone_tensor_for_ema(v) for k, v in state_dict.items()}


def _match_state_dict_prefix_to_reference(state_dict, reference_state_dict):
    reference_has_prefix = _state_dict_has_module_prefix(reference_state_dict)
    state_has_prefix = _state_dict_has_module_prefix(state_dict)
    if reference_has_prefix and not state_has_prefix:
        return _add_module_prefix(state_dict)
    if state_has_prefix and not reference_has_prefix:
        return _strip_module_prefix(state_dict)
    return state_dict


class EMACheckpoint:
    def __init__(
        self,
        *,
        ema_weight,
        save_interval,
        save_dir,
        config,
        state_dict_getter,
        logger,
    ):
        self.ema_weight = float(ema_weight)
        if not 0.0 <= self.ema_weight <= 1.0:
            raise ValueError("train.ema.ema_weight must be in [0, 1].")

        self.save_interval = int(save_interval)
        if self.save_interval <= 0:
            raise ValueError("train.ema.save_interval must be a positive integer.")

        self.config = config
        self.state_dict_getter = state_dict_getter
        self.logger = logger
        self.latest_path = os.path.join(save_dir, "ema_checkpoint.pt")
        self.final_path = os.path.join(save_dir, "ema_checkpoint_final.pt")
        self.state_dict = None
        self.last_update_step = None

    # initialize the ema checkpoint or resume
    def load_or_initialize(self, start_step):
        # resume
        if start_step > 0 and os.path.isfile(self.latest_path):
            checkpoint = _torch_load_checkpoint(self.latest_path, map_location="cpu")
            ema_step = int(checkpoint.get("step", 0))
            if ema_step <= start_step and "model_state_dict" in checkpoint:
                self._load_state_dict(checkpoint["model_state_dict"], ema_step)
                return

            self.logger.warning(
                "Skipping EMA checkpoint %s because its step=%d is newer than resume step=%d.",
                self.latest_path,
                ema_step,
                start_step,
            )
            
        # initialize
        self.initialize_from_current(start_step)

    def _load_state_dict(self, state_dict, step):
        reference_state_dict = self.state_dict_getter()
        state_dict = _match_state_dict_prefix_to_reference(state_dict, reference_state_dict)

        reference_keys = set(reference_state_dict.keys())
        loaded_keys = set(state_dict.keys())
        if reference_keys != loaded_keys:
            missing = len(reference_keys - loaded_keys)
            unexpected = len(loaded_keys - reference_keys)
            self.logger.warning(
                "EMA checkpoint key mismatch (missing=%d unexpected=%d); initializing EMA from current model.",
                missing,
                unexpected,
            )
            self.initialize_from_current(step)
            return

        self.state_dict = _clone_state_dict_for_ema(state_dict)
        self.last_update_step = int(step)
        self.logger.info("Loaded EMA weights at step %d", self.last_update_step)

    def initialize_from_current(self, step):
        self.state_dict = _clone_state_dict_for_ema(self.state_dict_getter())
        self.last_update_step = int(step)
        self.logger.info("Initialized EMA weights from current model at step %d", self.last_update_step)

    # save when reach interval
    def maybe_update_and_save(self, step):
        if step > 0 and step % self.save_interval == 0:
            self.update(step)
            self.save(self.latest_path, step)

    @torch.no_grad()
    def update(self, step):
        current_state_dict = self.state_dict_getter()
        if self.state_dict is None:
            self.state_dict = _clone_state_dict_for_ema(current_state_dict)
            self.last_update_step = int(step)
            return

        current_weight = 1.0 - self.ema_weight
        for key, current_value in current_state_dict.items():
            current_value = current_value.detach().cpu()
            ema_value = self.state_dict.get(key)
            if ema_value is None or ema_value.shape != current_value.shape:
                self.state_dict[key] = _clone_tensor_for_ema(current_value)
                continue

            if ema_value.is_floating_point() and current_value.is_floating_point():
                ema_value.mul_(self.ema_weight)
                ema_value.add_(current_value.to(dtype=ema_value.dtype), alpha=current_weight)
            else:
                ema_value.copy_(current_value.to(dtype=ema_value.dtype))

        self.last_update_step = int(step)

    def save(self, path, step):
        if self.state_dict is None:
            self.initialize_from_current(step)

        torch.save({
            'step': int(step),
            'model_state_dict': self.state_dict,
            'config': self.config,
            'checkpoint_type': 'ema',
            'ema_weight': self.ema_weight,
            'ema_save_interval': self.save_interval,
        }, path)
        self.logger.info("Saved EMA checkpoint to %s", path)

    # final save
    def finalize(self, step):
        if self.last_update_step != step:
            self.update(step)
        self.save(self.latest_path, step)
        self.save(self.final_path, step)

# -----------------------------------------------------------------------------
# --------------------------- Alignment helper --------------------------------
# -----------------------------------------------------------------------------
def _get_text_embedding(model):
    if hasattr(model, "get_text_embedding"):
        return model.get_text_embedding()

    lmm = getattr(model, "lmm", None)
    if lmm is not None and hasattr(lmm, "get_input_embeddings"):
        return lmm.get_input_embeddings()

    backbone = getattr(lmm, "model", None)
    if backbone is not None and hasattr(backbone, "get_input_embeddings"):
        return backbone.get_input_embeddings()

    return None


def _get_vision_encoder(model):
    if hasattr(model, "get_vision_encoder"):
        return model.get_vision_encoder()

    family = getattr(model, "model_family", None)
    if family == "llama":
        return getattr(model, "vision_encoder", None)
    if hasattr(model, "vision_encoder"):
        return model.vision_encoder

    lmm_model = getattr(getattr(model, "lmm", None), "model", None)
    if lmm_model is None:
        return None
    if family == "qwen":
        return getattr(lmm_model, "visual", None)
    if family == "paligemma":
        return getattr(lmm_model, "vision_tower", None)
    return None


def _format_module_trainable(module):
    if module is None:
        return "N/A"
    return "ON" if any(p.requires_grad for p in module.parameters()) else "OFF"


def _set_alignment_trainable(model, train_action_modules=True, train_full_policy=False, train_text_embedding=False, train_vision_encoder=False):
    for p in model.parameters():
        p.requires_grad = False
    if train_action_modules:
        if getattr(model, "action_projector", None) is not None:
            for p in model.action_projector.parameters():
                p.requires_grad = True
        if hasattr(model, "action_head"):
            if train_full_policy:
                for p in model.action_head.parameters():
                    p.requires_grad = True
            elif hasattr(model.action_head, "final_layer"):
                for p in model.action_head.final_layer.parameters():
                    p.requires_grad = True
    if train_text_embedding:
        text_embedding = _get_text_embedding(model)
        if text_embedding is not None:
            text_embedding.requires_grad_(True)
    if train_vision_encoder:
        vision_encoder = _get_vision_encoder(model)
        if vision_encoder is not None:
            vision_encoder.requires_grad_(True)


def _restore_trainable_after_alignment(
    model,
    backbone_mode,
    train_text_embedding=None,
    train_vision_encoder=None,
):
    for p in model.parameters():
        p.requires_grad = True

    model.configure_backbone_trainability(
        backbone_mode,
        train_text_embedding=train_text_embedding,
        train_vision_encoder=train_vision_encoder,
    )

    if getattr(model, "vq_model", None) is not None:
        model.vq_model.requires_grad_(False)
        model.vq_model.eval()
    if getattr(model, "dino_model", None) is not None:
        model.dino_model.requires_grad_(False)
        model.dino_model.eval()


def _reinit_action_modules(model):
    if getattr(model, "action_projector", None) is not None:
        ap = model.action_projector
        if hasattr(ap, "initialize_weights"):
            ap.initialize_weights()
        else:
            for m in ap.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.LayerNorm):
                    if m.elementwise_affine:
                        torch.nn.init.ones_(m.weight)
                        torch.nn.init.zeros_(m.bias)
    if hasattr(model.action_head, "final_layer"):
        fl = model.action_head.final_layer
        # Match the zero-init used by the policy module's initialize_weights().
        torch.nn.init.constant_(fl.adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(fl.adaLN_modulation[-1].bias, 0)
        torch.nn.init.constant_(fl.linear.weight, 0)
        torch.nn.init.constant_(fl.linear.bias, 0)

# -----------------------------------------------------------------------------
# ---------------------------- Training Loop ----------------------------------
# -----------------------------------------------------------------------------
def _run_training_loop(
    *,
    train_model,
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    config,
    save_dir,
    device,
    gradient_accumulation_steps,
    is_distributed,
    use_deepspeed,
    model_engine,
    global_rank,
    max_steps,
    start_step,
    stage_desc,
    log_prefix,
    ds_save_prefix,
    file_save_prefix,
    enable_save,
    logger,
    ema_checkpoint=None,
):
    """Inner training loop shared between alignment and action-generation stages."""
    train_model.train()
    unwrapped = model.module if hasattr(model, "module") else model
    if getattr(unwrapped, "vq_model", None) is not None:
        unwrapped.vq_model.eval()
    if getattr(unwrapped, "dino_model", None) is not None:
        unwrapped.dino_model.eval()
    step = start_step
    batch_idx = 0
    use_progress_bar = global_rank == 0 and sys.stderr.isatty()
    progress_bar = tqdm(total=max_steps, initial=start_step, desc=stage_desc) if use_progress_bar else None
    data_iter = iter(dataloader)
    if not use_deepspeed:
        optimizer.zero_grad()

    log_interval = config['project']['log_interval']
    save_interval = config['project']['save_interval']
    use_wandb = config['project'].get('use_wandb', False)
    max_grad_norm = config['train'].get('max_grad_norm', 1.0)
    stage_start_time = time.time()
    last_log_time = stage_start_time
    last_log_step = start_step
    remaining_steps = max(0, max_steps - start_step)

    if global_rank == 0:
        logger.info(
            "Starting stage=%s start_step=%d max_steps=%d remaining_steps=%d log_interval=%d save_interval=%d",
            log_prefix,
            start_step,
            max_steps,
            remaining_steps,
            log_interval,
            save_interval,
        )

    def move_input_dict(input_dict):
        if input_dict is None:
            return None
        moved = {}
        for k, v in input_dict.items():
            if torch.is_tensor(v):
                dtype = torch.bfloat16 if k in ['pixel_values', 'pixel_values_videos'] else None
                moved[k] = v.to(device=device, dtype=dtype, non_blocking=True)
            else:
                moved[k] = v
        return moved

    metric_sums = {}
    metric_count = 0

    def unpack_loss_output(output):
        if torch.is_tensor(output):
            return output, {"loss": output.detach().float()}
        if not isinstance(output, dict) or "loss" not in output:
            raise TypeError("Model forward must return a loss tensor or a dict containing `loss`.")

        loss = output["loss"]
        metrics = {"loss": loss.detach().float()}
        for group_name in ("loss_components", "weighted_loss_components"):
            for name, value in (output.get(group_name) or {}).items():
                if value is None:
                    continue
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=loss.device)
                metrics[f"{group_name}/{name}"] = value.detach().float()
        return loss, metrics

    def accumulate_loss_metrics(metrics):
        nonlocal metric_count
        for name, value in metrics.items():
            if value.ndim > 0:
                value = value.mean()
            value = value.detach().float()
            if name not in metric_sums:
                metric_sums[name] = value.clone()
            else:
                metric_sums[name] = metric_sums[name] + value
        metric_count += 1

    def flush_loss_metrics(reduce_across_ranks=False, return_scalars=True):
        nonlocal metric_sums, metric_count
        if metric_count == 0:
            return {}

        metrics = {name: value / metric_count for name, value in metric_sums.items()}
        if reduce_across_ranks and is_distributed and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for name in sorted(metrics):
                dist.all_reduce(metrics[name], op=dist.ReduceOp.SUM)
                metrics[name] = metrics[name] / world_size

        result = {name: value.item() for name, value in metrics.items()} if return_scalars else {}
        metric_sums = {}
        metric_count = 0
        return result

    def log_training_progress(loss_value):
        nonlocal last_log_time, last_log_step
        now = time.time()
        completed = step - start_step
        total = max_steps - start_step
        elapsed, avg_step_time, eta = progress_timing(stage_start_time, completed, total, now=now)
        recent_steps = max(1, step - last_log_step)
        recent_step_time = (now - last_log_time) / recent_steps
        last_log_time = now
        last_log_step = step
        progress = (step / max_steps * 100.0) if max_steps > 0 else 100.0
        logger.info(
            "stage=%s step=%d/%d progress=%.1f%% loss=%.6f lr=%.6e elapsed=%s avg_step=%.2fs recent_step=%.2fs eta=%s",
            log_prefix,
            step,
            max_steps,
            progress,
            loss_value,
            lr_scheduler.get_last_lr()[0],
            format_duration(elapsed),
            avg_step_time,
            recent_step_time,
            format_duration(eta),
        )

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if len(batch) == 5:
            inputs, gt_actions, proprio, hist_actions, future_images = batch
            future_videos = None
        else:
            inputs, gt_actions, proprio, hist_actions, future_images, future_videos = batch
        del batch
        model_inputs = move_input_dict(inputs)
        del inputs

        gt_actions = gt_actions.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        if proprio is not None:
            proprio = proprio.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        if hist_actions is not None:
            hist_actions = hist_actions.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        if future_images is not None:
            future_images = future_images.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        if future_videos is not None:
            future_videos = future_videos.to(device=device, dtype=torch.bfloat16, non_blocking=True)

        valid_keys = {
            "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw", "mm_token_type_ids", "token_type_ids",
            "language_action_labels", "prompt_texts", "current_images",
        }
        forward_args = {k: v for k, v in model_inputs.items() if k in valid_keys}

        if use_deepspeed:
            output = model_engine(
                actions=gt_actions,
                proprioception=proprio,
                history_actions=hist_actions,
                future_images=future_images,
                future_videos=future_videos,
                return_loss_components=True,
                **forward_args
            )
            loss, loss_metrics = unpack_loss_output(output)
            accumulate_loss_metrics(loss_metrics)
            is_boundary = model_engine.is_gradient_accumulation_boundary()
            model_engine.backward(loss)
            model_engine.step()

            batch_idx += 1
            if is_boundary:
                step += 1
                is_log_step = step % log_interval == 0 or step == max_steps
                should_log = is_log_step and global_rank == 0
                metrics = flush_loss_metrics(
                    reduce_across_ranks=is_log_step,
                    return_scalars=should_log,
                )
                loss_value = metrics.get("loss") if should_log else None
                if progress_bar is not None:
                    progress_bar.update(1)
                if should_log:
                    log_training_progress(loss_value)
                    if use_wandb:
                        wandb_payload = {f"{log_prefix}/{name}": value for name, value in metrics.items()}
                        wandb_payload.update({
                            f"{log_prefix}/lr": lr_scheduler.get_last_lr()[0],
                            "step": step,
                        })
                        wandb.log(wandb_payload)
                if enable_save and step % save_interval == 0:
                    ds_save_dir = os.path.join(save_dir, f"{ds_save_prefix}_{step}")
                    model_engine.save_checkpoint(ds_save_dir, client_state={'step': step, 'config': config})
                    if global_rank == 0:
                        logger.info("Saved DeepSpeed checkpoint to %s", ds_save_dir)
                if ema_checkpoint is not None:
                    ema_checkpoint.maybe_update_and_save(step)
        else:
            do_update = (batch_idx + 1) % gradient_accumulation_steps == 0
            sync_context = model.no_sync if (is_distributed and not do_update) else nullcontext
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        actions=gt_actions,
                        proprioception=proprio,
                        history_actions=hist_actions,
                        future_images=future_images,
                        future_videos=future_videos,
                        return_loss_components=True,
                        **forward_args
                    )
                loss, loss_metrics = unpack_loss_output(output)
                accumulate_loss_metrics(loss_metrics)
                loss = loss / gradient_accumulation_steps
                loss.backward()

            if do_update:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                is_log_step = step % log_interval == 0 or step == max_steps
                should_log = is_log_step and global_rank == 0
                metrics = flush_loss_metrics(
                    reduce_across_ranks=is_log_step,
                    return_scalars=should_log,
                )
                loss_value = metrics.get("loss") if should_log else None
                if progress_bar is not None:
                    progress_bar.update(1)
                if should_log:
                    log_training_progress(loss_value)
                    if use_wandb:
                        wandb_payload = {f"{log_prefix}/{name}": value for name, value in metrics.items()}
                        wandb_payload.update({
                            f"{log_prefix}/lr": lr_scheduler.get_last_lr()[0],
                            "step": step,
                        })
                        wandb.log(wandb_payload)
                if enable_save and step % save_interval == 0 and global_rank == 0:
                    save_path = os.path.join(save_dir, f"{file_save_prefix}_{step}.pt")
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'config': config
                    }, save_path)
                    logger.info("Saved checkpoint to %s", save_path)
                if ema_checkpoint is not None:
                    ema_checkpoint.maybe_update_and_save(step)

            batch_idx += 1

    if progress_bar is not None:
        progress_bar.close()
    if global_rank == 0:
        elapsed, avg_step_time, _ = progress_timing(
            stage_start_time,
            max(0, step - start_step),
            max(0, max_steps - start_step),
        )
        logger.info(
            "Finished stage=%s final_step=%d/%d elapsed=%s avg_step=%.2fs",
            log_prefix,
            step,
            max_steps,
            format_duration(elapsed),
            avg_step_time,
        )
    return step


def train(config):
    # -----------------------------------------------------------------------------
    # ----------------------------------- Setup -----------------------------------
    # -----------------------------------------------------------------------------
    atexit.register(_cleanup_on_exit)
    use_deepspeed = config['train'].get('deepspeed', {}).get('enabled', False)
    is_distributed = config['train'].get('distributed', False) or use_deepspeed
    gradient_accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
    use_proprio_input_vlm = config['model'].get('use_proprio_input_vlm', True)
    use_action_input_policy = config['model'].get('use_action_input_policy', True)
    lmm_path = config['model']['lmm_path']
    use_wan_backbone = "wan" in str(lmm_path).lower()
    future_image_loss_weight = float(config['model'].get('future_image_loss_weight', 0.0))
    language_action_loss_weight = float(config['model'].get('language_action_loss_weight', 0.0))
    if use_wan_backbone:
        future_image_loss_weight = 0.0
        language_action_loss_weight = 0.0
    load_future_image = future_image_loss_weight > 0
    load_future_video = use_wan_backbone
    future_video_downsample = int(config['data'].get('future_video_downsample', config['model'].get('future_video_downsample', 1)))
    future_image_prediction_type = config['model'].get('future_image_prediction_type', 'emu_token')
    future_image_mode = config['model'].get('future_image_mode', 'horizon')
    load_language_action = language_action_loss_weight > 0
    language_action_format = normalize_language_action_format(
        config['model'].get('language_action_format', 'lap')
    )
    language_action_num_bins = int(config['model'].get('language_action_num_bins', 1000))
    input_modality = config["data"].get("input_modality", "video")
    view_mode = config["data"].get("view_mode", "single")
    augmentation = config["data"].get("augmentation", {})
    dataset_name = config['data'].get('dataset_name', 'libero')
    dataset_format = config['data'].get('dataset_format', 'tfds')
    supported_dataset_names = {"libero", "real", "droid"}
    if dataset_name not in supported_dataset_names:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Supported datasets: {', '.join(sorted(supported_dataset_names))}."
        )
    if load_future_video and dataset_name not in {"libero", "droid"}:
        raise ValueError("WAN video-generation backbone currently requires a LeRobot robot dataset loader.")
    if dataset_name == "libero":
        supported_dataset_formats = {"tfds", "lerobot"}
        if dataset_format not in supported_dataset_formats:
            raise ValueError(
                f"Unsupported LIBERO dataset_format '{dataset_format}'. "
                f"Supported formats: {', '.join(sorted(supported_dataset_formats))}."
            )
    elif dataset_name == "droid" and dataset_format != "lerobot":
        raise ValueError("Droid only supports dataset_format='lerobot'.")
    if load_language_action and dataset_name not in {"libero", "droid"} and language_action_format != "vla-0":
        raise ValueError(
            "language_action_format='lap' is only implemented for LIBERO and Droid. "
            "Use language_action_format='vla-0' for other datasets."
        )
    pretrained_ckpt_path = config['train'].get('pretrained_checkpoint')
    bimanual = config['data'].get('bimanual', False)
    allow_end_padding = config['data'].get('allow_end_padding', True)
    image_resize_size = config['data'].get('image_resize_size', None)
    if dataset_name == "libero":
        config['data']['normalization_suite_name'] = get_libero_normalization_suite_name(
            config['data'].get('task_suite_name'),
            configured_suite_name=config['data'].get('normalization_suite_name'),
        )
    if dataset_name == "libero":
        fps = 20.0
    elif dataset_name == "real":
        fps = 5.0
    elif dataset_name == "droid":
        fps = float(config["data"].get("fps", 15.0))
    else:
        fps = 20.0
    full_sequence = bool(config['data'].get('full_sequence', False))
    action_mode = str(config['data'].get('action_mode', 'libero')).lower()
    seed = config['train'].get('seed', 42)

    set_seed(seed)

    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed(dist_backend=config['train'].get('dist_backend', 'nccl'))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif is_distributed:
        dist.init_process_group(backend=config['train']['dist_backend'])
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device(config['train'].get('device', 'cuda'))
        
    wandb_project = config['project'].get('wandb_project', 'VLANeXt')
    wandb_name = config['project']['name']
    full_name = config['project']['name']
    parts = full_name.split('_')
    if len(parts) > 3:
        parent_dir = '_'.join(parts[:3])
        sub_dir = '_'.join(parts[3:])
        save_dir = os.path.join(config['project']['output_dir'], parent_dir, sub_dir)
        wandb_project = parent_dir
        wandb_name = sub_dir
    else:
        save_dir = os.path.join(config['project']['output_dir'], full_name)
    if global_rank == 0:
        os.makedirs(config['project']['output_dir'], exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(
        "train",
        os.path.join(save_dir, "train.log") if global_rank == 0 else None,
        rank=global_rank,
    )
    if global_rank == 0:
        logger.info(
            "Starting training project=%s save_dir=%s distributed=%s deepspeed=%s world_size=%d device=%s",
            full_name,
            save_dir,
            is_distributed,
            use_deepspeed,
            world_size,
            device,
        )
    if config['project'].get('use_wandb', False) and global_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=config['project'].get('wandb_entity', None),
            name=wandb_name,
            config=config
        )

    has_pretrained_ckpt = (pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path))
    main_train_text_embedding = config['model'].get('train_text_embedding', None)
    main_train_vision_encoder = config['model'].get('train_vision_encoder', None)
    model = VLANeXt(
        lmm_path=lmm_path,
        vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
        use_pretrained_backbone=config['model'].get('use_pretrained_backbone', True),
        action_dim=config['model']['action_dim'],
        num_actions=config['data']['future_len'],
        num_queries=config['model']['num_queries'],
        num_history=config['data']['history_len'],
        loss_type=config['model'].get('loss_type', 'diffusion'),
        future_image_loss_weight=future_image_loss_weight,
        num_train_timesteps=config['model'].get('num_train_timesteps', 1000),
        num_inference_timesteps=config['model'].get('num_inference_timesteps', 10),
        scheduler_type=config['model']['scheduler_type'],
        diffusion_loss_domain=config['model'].get('diffusion_loss_domain', 'noise'),
        condition_type=config['model'].get('condition_type', 'loose'),
        policy_hidden_size=config['model']['policy_hidden_size'],
        policy_depth=config['model']['policy_depth'],
        policy_num_heads=config['model']['policy_num_heads'],
        policy_mlp_ratio=config['model']['policy_mlp_ratio'],
        use_proprio_input_vlm=use_proprio_input_vlm,
        use_action_input_policy=use_action_input_policy,
        use_transformer_proprio_projector=config['model']['use_transformer_proprio_projector'],
        projector_depth=config['model']['projector_depth'],
        projector_num_heads=config['model']['projector_num_heads'],
        use_transformer_connector=config['model']['use_transformer_connector'],
        connector_depth=config['model']['connector_depth'],
        connector_num_heads=config['model']['connector_num_heads'],
        backbone_mode=config['model'].get('backbone_mode', 'finetune'),
        train_text_embedding=main_train_text_embedding,
        train_vision_encoder=main_train_vision_encoder,
        gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
        num_bins=config['model'].get('num_bins', 256),
        classification_type=config['model'].get('classification_type', 'parallel'),
        fast_action_tokenizer=config['model'].get('fast_action_tokenizer', None),
        generator_hidden_size=config['model'].get('generator_hidden_size', 768),
        generator_depth=config['model'].get('generator_depth', config['model'].get('policy_depth', 12)),
        generator_num_heads=config['model'].get('generator_num_heads', 12),
        generator_mlp_ratio=config['model'].get('generator_mlp_ratio', 4.0),
        generator_max_seq_len=config['model'].get('generator_max_seq_len', 1024),
        future_image_num_tokens=config['model'].get('future_image_num_tokens', 256),
        future_image_prediction_type=future_image_prediction_type,
        future_image_dino_model_path=config['model'].get(
            'future_image_dino_model_path',
            'facebook/dinov3-vitb16-pretrain-lvd1689m',
        ),
        future_image_dino_image_size=config['model'].get('future_image_dino_image_size', 256),
        future_image_flow_num_inference_timesteps=config['model'].get(
            'future_image_flow_num_inference_timesteps',
            10,
        ),
        language_action_loss_weight=language_action_loss_weight,
        policy_gradient_stop_for_vlm=config['model'].get('policy_gradient_stop_for_vlm', False),
        dct_loss_weight=config['model'].get('dct_loss_weight', 0.1),
        dct_low_freq_weight=config['model'].get('dct_low_freq_weight', 1.0),
        dct_high_freq_weight=config['model'].get('dct_high_freq_weight', 1.0),
        dct_freq_split=config['model'].get('dct_freq_split', 0.125),
        dct_similarity_type=config['model'].get('dct_similarity_type', 'mae'),
        video_generation_loss_weight=config['model'].get('video_generation_loss_weight', 1.0),
        wan_action_condition_mode=config['model'].get('wan_action_condition_mode', 'fast'),
        wan_flow_shift=config['model'].get('wan_flow_shift', 5.0),
        wan_text_len=config['model'].get('wan_text_len', 512),
        wan_tokenizer_model_id=config['model'].get('wan_tokenizer_model_id', 'google/umt5-xxl'),
        future_video_downsample=future_video_downsample,
    ).to(device)
    if has_pretrained_ckpt:
        if global_rank == 0:
            logger.info("Loading pretrained VLA checkpoint: %s", pretrained_ckpt_path)
        state_dict, checkpoint_info = _load_model_state_dict_from_checkpoint(
            pretrained_ckpt_path,
            map_location=device,
        )
        state_dict = _strip_module_prefix(state_dict)
        if config['train'].get('pretrained_ignore_mismatched_shapes', False):
            state_dict, _ = _drop_mismatched_state_dict_entries(
                model,
                state_dict,
                logger=logger if global_rank == 0 else None,
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if global_rank == 0:
            if checkpoint_info["type"] == "deepspeed":
                logger.info(
                    "Loaded DeepSpeed pretrained weights from %s tag=%s step=%d",
                    checkpoint_info["load_dir"],
                    checkpoint_info["tag"],
                    checkpoint_info["step"],
                )
            logger.info("Loaded weights. missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        if global_rank == 0:
            if pretrained_ckpt_path:
                logger.warning("Pretrained checkpoint path does not exist; training from scratch: %s", pretrained_ckpt_path)
            else:
                logger.info("No pretrained checkpoint provided; training from scratch")

    if use_deepspeed:
        model_unwrapped = model
    elif is_distributed:
        # find_unused_parameters=True is needed when alignment is enabled because
        # the trainable param set changes between alignment and action-generation stages.
        align_enabled = config['train'].get('train_alignment', {}).get('enabled', False)
        ddp_kwargs = {'device_ids': [local_rank]}
        if align_enabled:
            ddp_kwargs['find_unused_parameters'] = True
        model = DDP(model, **ddp_kwargs)
        model_unwrapped = model.module
    else:
        model_unwrapped = model

    if global_rank == 0:
        should_exit = run_size_speed_eval(
            model_unwrapped, config, device, torch.bfloat16,
            input_modality, view_mode, bimanual, fps
        )
        if should_exit:
            return

    # -----------------------------------------------------------------------------
    # -------------------------------- Data Loader --------------------------------
    # -----------------------------------------------------------------------------
    data_root = config['data']['data_root']
    if dataset_name == "real":
        task_suite = config['data']['task_suite_name']
        real_world_path = os.path.join(data_root, task_suite)
        if global_rank == 0:
            logger.info("Initializing RealWorld Dataset: task=%s root=%s", task_suite, data_root)
    elif dataset_name == "droid":
        task_suite = config['data'].get('task_suite_name', 'droid')
        droid_path = data_root
        if global_rank == 0:
            logger.info("Initializing Droid LeRobot Dataset: path=%s fps=%.2f", droid_path, fps)
    else:
        task_suite = config['data']['task_suite_name']
        stats_suite = config['data'].get('normalization_suite_name', task_suite)
        if task_suite == "libero_mixed":
            if global_rank == 0:
                logger.info(
                    "Initializing Libero Mixed Dataset: format=%s root=%s normalization_stats=%s",
                    dataset_format,
                    data_root,
                    stats_suite,
                )
        else:
            if dataset_format == "lerobot":
                libero_path = os.path.join(data_root, task_suite)
            else:
                libero_path = os.path.join(data_root, task_suite, "1.0.0")
            if global_rank == 0:
                logger.info(
                    "Initializing Libero Dataset: format=%s path=%s normalization_stats=%s",
                    dataset_format,
                    libero_path,
                    stats_suite,
                )
    collator = DataCollatorForVLANeXt(
        processor=model_unwrapped.processor,
        use_proprio_input_vlm=use_proprio_input_vlm,
        use_action_input_policy=use_action_input_policy,
        input_modality=input_modality,
        view_mode=view_mode,
        fps=fps,
        augmentation=augmentation,
        bimanual=bimanual,
        load_future_image=load_future_image,
        future_image_prediction_type=future_image_prediction_type,
        future_image_processor=getattr(model_unwrapped, "future_image_processor", None),
        load_language_action=load_language_action,
        language_action_format=language_action_format,
        language_action_num_bins=language_action_num_bins,
        load_future_video=load_future_video,
        wan_prompt_template=config['model'].get('wan_prompt_template', DEFAULT_WAN_PROMPT_TEMPLATE),
        future_len=config['data']['future_len'],
        action_dim=config['model']['action_dim'],
    )
    total_batch_size = config['data']['batch_size']
    per_device_batch_size = total_batch_size // (world_size * gradient_accumulation_steps)
    if global_rank == 0:
        logger.info(
            "Batch config: total_batch_size=%d world_size=%d grad_acc_steps=%d per_device_batch_size=%d",
            total_batch_size,
            world_size,
            gradient_accumulation_steps,
            per_device_batch_size,
        )
        logger.info("Dataset image_resize_size=%s", image_resize_size)

    buffer_size = config['data'].get("buffer_size", 10000)
    if global_rank == 0:
        if (dataset_name == "libero" and dataset_format == "lerobot") or dataset_name == "droid":
            logger.info(
                "LeRobot dataset emits prebatched random samples: batch_size=%d drop_last=%s",
                per_device_batch_size,
                bool(config['data'].get('drop_last', True)),
            )
        else:
            logger.info("Dataset shuffle buffer size: %d", buffer_size)

    def create_dataloader(history_len):
        if dataset_name == "real":
            ds = RealWorldAct(
                data_path=real_world_path,
                bimanual=bimanual,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                buffer_size=buffer_size,
                sampling_rate=config['data'].get('sampling_rate', 0.1),
                episode_downsample_factor=config['data'].get('episode_downsample_factor', 1),
                allow_end_padding=allow_end_padding,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                dataset_name=task_suite,
                seed=seed,
            )
        elif dataset_name == "droid":
            ds = DroidLeRobotAct(
                data_path=droid_path,
                dataset_name=task_suite,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                load_future_video=load_future_video,
                future_video_downsample=future_video_downsample,
                buffer_size=buffer_size,
                sampling_rate=config['data'].get('sampling_rate', 0.1),
                allow_end_padding=allow_end_padding,
                emit_proprioception=use_proprio_input_vlm,
                emit_history_actions=use_action_input_policy,
                batch_size=per_device_batch_size,
                drop_last=bool(config['data'].get('drop_last', True)),
                episode_cache_size=int(config['data'].get('episode_cache_size', 64)),
                data_file_cache_size=int(config['data'].get('data_file_cache_size', 2)),
                seed=seed,
                image_resize_size=image_resize_size,
                load_language_action=load_language_action,
                language_action_format=language_action_format,
                language_action_num_bins=language_action_num_bins,
                action_mode=action_mode,
                fps=fps,
            )
        else:
            if dataset_format == "lerobot":
                if task_suite == "libero_mixed":
                    ds = LiberoMixedLeRobotAct(
                        data_root=data_root,
                        normalization_suite_name=stats_suite,
                        history_len=history_len,
                        future_len=config['data']['future_len'],
                        full_sequence=full_sequence,
                        input_modality=input_modality,
                        view_mode=view_mode,
                        load_future_image=load_future_image,
                        future_image_mode=future_image_mode,
                        load_future_video=load_future_video,
                        future_video_downsample=future_video_downsample,
                        buffer_size=buffer_size,
                        sampling_rate=config['data'].get('sampling_rate', 0.1),
                        allow_end_padding=allow_end_padding,
                        emit_proprioception=use_proprio_input_vlm,
                        emit_history_actions=use_action_input_policy,
                        batch_size=per_device_batch_size,
                        drop_last=bool(config['data'].get('drop_last', True)),
                        seed=seed,
                        image_resize_size=image_resize_size,
                        load_language_action=load_language_action,
                        language_action_format=language_action_format,
                        language_action_num_bins=language_action_num_bins,
                        action_mode=action_mode,
                    )
                else:
                    ds = LiberoLeRobotAct(
                        data_path=libero_path,
                        dataset_name=task_suite,
                        normalization_suite_name=stats_suite,
                        history_len=history_len,
                        future_len=config['data']['future_len'],
                        full_sequence=full_sequence,
                        input_modality=input_modality,
                        view_mode=view_mode,
                        load_future_image=load_future_image,
                        future_image_mode=future_image_mode,
                        load_future_video=load_future_video,
                        future_video_downsample=future_video_downsample,
                        buffer_size=buffer_size,
                        sampling_rate=config['data'].get('sampling_rate', 0.1),
                        allow_end_padding=allow_end_padding,
                        emit_proprioception=use_proprio_input_vlm,
                        emit_history_actions=use_action_input_policy,
                        batch_size=per_device_batch_size,
                        drop_last=bool(config['data'].get('drop_last', True)),
                        seed=seed,
                        image_resize_size=image_resize_size,
                        load_language_action=load_language_action,
                        language_action_format=language_action_format,
                        language_action_num_bins=language_action_num_bins,
                        action_mode=action_mode,
                    )
            elif task_suite == "libero_mixed":
                ds = LiberoMixedAct(
                    data_root=data_root,
                    history_len=history_len,
                    future_len=config['data']['future_len'],
                    full_sequence=full_sequence,
                    input_modality=input_modality,
                    view_mode=view_mode,
                    load_future_image=load_future_image,
                    future_image_mode=future_image_mode,
                    load_future_video=load_future_video,
                    future_video_downsample=future_video_downsample,
                    buffer_size=buffer_size,
                    sampling_rate=config['data'].get('sampling_rate', 0.1),
                    allow_end_padding=allow_end_padding,
                    emit_proprioception=use_proprio_input_vlm,
                    emit_history_actions=use_action_input_policy,
                    image_resize_size=image_resize_size,
                    load_language_action=load_language_action,
                    language_action_format=language_action_format,
                    language_action_num_bins=language_action_num_bins,
                )
            else:
                ds = LiberoAct(
                    data_path=libero_path,
                    dataset_name=task_suite,
                    normalization_suite_name=stats_suite,
                    history_len=history_len,
                    future_len=config['data']['future_len'],
                    full_sequence=full_sequence,
                    input_modality=input_modality,
                    view_mode=view_mode,
                    load_future_image=load_future_image,
                    future_image_mode=future_image_mode,
                    load_future_video=load_future_video,
                    future_video_downsample=future_video_downsample,
                    buffer_size=buffer_size,
                    sampling_rate=config['data'].get('sampling_rate', 0.1),
                    allow_end_padding=allow_end_padding,
                    emit_proprioception=use_proprio_input_vlm,
                    emit_history_actions=use_action_input_policy,
                    image_resize_size=image_resize_size,
                    load_language_action=load_language_action,
                    language_action_format=language_action_format,
                    language_action_num_bins=language_action_num_bins,
                )
        num_workers = int(config['data'].get('num_workers', 0))
        if (dataset_name == "libero" and dataset_format == "lerobot") or dataset_name == "droid":
            loader_kwargs = {
                'batch_size': None,
                'num_workers': num_workers,
                'collate_fn': collator,
                'pin_memory': bool(config['data'].get('pin_memory', True)),
            }
        else:
            loader_kwargs = {
                'batch_size': per_device_batch_size,
                'num_workers': num_workers,
                'collate_fn': collator,
                'pin_memory': bool(config['data'].get('pin_memory', True)),
                'drop_last': bool(config['data'].get('drop_last', True)),
            }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = bool(config['data'].get('persistent_workers', True))
            loader_kwargs['prefetch_factor'] = int(config['data'].get('prefetch_factor', 4))
        return DataLoader(ds, **loader_kwargs)

    dataloader = create_dataloader(config['data']['history_len'])

    # -----------------------------------------------------------------------------
    # ---------------------------- Alignment Training -----------------------------
    # -----------------------------------------------------------------------------
    align_cfg = config['train'].get('train_alignment', {}) or {}
    has_resume = bool(config['train'].get('resume_path')) and os.path.exists(config['train'].get('resume_path', ''))
    do_alignment = bool(align_cfg.get('enabled', False)) and not has_resume

    if do_alignment:
        backbone_mode = config['model'].get('backbone_mode', 'finetune')
        if global_rank == 0:
            logger.info("Starting alignment training stage")

        align_train_action_modules = bool(align_cfg.get('train_action_modules', True))
        align_train_full_policy = bool(align_cfg.get('train_full_policy', False))
        align_reinit_action_modules = bool(align_cfg.get('reinit_action_modules', False))
        align_train_text_embedding = bool(align_cfg.get('train_text_embedding', False))
        align_train_vision_encoder = bool(align_cfg.get('train_vision_encoder', False))

        if align_reinit_action_modules and not align_train_action_modules:
            raise ValueError(
                "train_alignment.reinit_action_modules=True requires train_alignment.train_action_modules=True."
            )

        if align_train_full_policy and not align_train_action_modules:
            raise ValueError(
                "train_alignment.train_full_policy=True requires train_alignment.train_action_modules=True."
            )

        if align_reinit_action_modules:
            if global_rank == 0:
                logger.info("Reinitializing action_projector and action_head.final_layer")
            _reinit_action_modules(model_unwrapped)

        _set_alignment_trainable(
            model_unwrapped,
            train_action_modules=align_train_action_modules,
            train_full_policy=align_train_full_policy,
            train_text_embedding=align_train_text_embedding,
            train_vision_encoder=align_train_vision_encoder,
        )
        if global_rank == 0:
            n_train = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model_unwrapped.parameters())
            am_scope = "full_policy" if align_train_full_policy else "final_layer_only"
            am_status = f"ON ({am_scope})" if align_train_action_modules else "OFF"
            te_status = "ON" if align_train_text_embedding else "OFF"
            ve_status = "ON" if align_train_vision_encoder else "OFF"
            logger.info(
                "Alignment trainable params: %s / %s (action_modules=%s text_embedding=%s vision_encoder=%s)",
                f"{n_train:,}",
                f"{n_total:,}",
                am_status,
                te_status,
                ve_status,
            )

        align_lr = float(align_cfg['learning_rate'])
        align_steps = int(align_cfg['alignment_steps'])
        align_warmup = int(align_cfg['warmup_steps'])

        align_engine = None
        if use_deepspeed:
            ds_config_align = build_deepspeed_config(
                config,
                per_device_batch_size,
                learning_rate=align_lr,
                warmup_steps=align_warmup,
                total_steps=align_steps,
            )
            align_engine, align_optimizer, _, align_scheduler = deepspeed.initialize(
                model=model,
                model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
                config=ds_config_align,
            )
            align_train_model = align_engine
        else:
            align_optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=align_lr,
                weight_decay=float(config['train']['weight_decay'])
            )
            align_scheduler = build_torch_lr_scheduler(
                get_lr_scheduler_name(config),
                optimizer=align_optimizer,
                num_warmup_steps=align_warmup,
                num_training_steps=align_steps,
            )
            align_train_model = model

        _run_training_loop(
            train_model=align_train_model,
            model=model,
            dataloader=dataloader,
            optimizer=align_optimizer,
            lr_scheduler=align_scheduler,
            config=config,
            save_dir=save_dir,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            is_distributed=is_distributed,
            use_deepspeed=use_deepspeed,
            model_engine=align_engine,
            global_rank=global_rank,
            max_steps=align_steps,
            start_step=0,
            stage_desc="Alignment",
            log_prefix="alignment",
            ds_save_prefix="align_ds_checkpoint",
            file_save_prefix="align_checkpoint",
            enable_save=False,
            logger=logger,
        )

        # Save final alignment checkpoint
        if use_deepspeed:
            align_save_dir = os.path.join(save_dir, "align_ds_checkpoint_final")
            align_engine.save_checkpoint(align_save_dir, client_state={'step': align_steps, 'stage': 'alignment'})
            if global_rank == 0:
                logger.info("Saved final alignment DeepSpeed checkpoint to %s", align_save_dir)
        else:
            if global_rank == 0:
                align_save_path = os.path.join(save_dir, "align_checkpoint_final.pt")
                torch.save({
                    'step': align_steps,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'stage': 'alignment',
                }, align_save_path)
                logger.info("Saved final alignment checkpoint to %s", align_save_path)

        # Tear down alignment optimizer/engine and unfreeze for the action-generation stage.
        del align_optimizer, align_scheduler
        if use_deepspeed:
            del align_engine
            torch.cuda.empty_cache()

        _restore_trainable_after_alignment(
            model_unwrapped,
            backbone_mode,
            train_text_embedding=main_train_text_embedding,
            train_vision_encoder=main_train_vision_encoder,
        )
        if global_rank == 0:
            logger.info("Alignment training stage finished")

    # -----------------------------------------------------------------------------
    # ------------------------ Action Generation Training -------------------------
    # -----------------------------------------------------------------------------
    start_step = 0
    if global_rank == 0:
        n_train = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model_unwrapped.parameters())
        logger.info(
            "Finetuning trainable params: %s / %s (text_embedding=%s vision_encoder=%s)",
            f"{n_train:,}",
            f"{n_total:,}",
            _format_module_trainable(_get_text_embedding(model_unwrapped)),
            _format_module_trainable(_get_vision_encoder(model_unwrapped)),
        )

    if use_deepspeed:
        ds_config = build_deepspeed_config(config, per_device_batch_size)
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
            config=ds_config,
        )
        device = model_engine.local_rank
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config['train']['learning_rate']),
            weight_decay=float(config['train']['weight_decay'])
        )
        lr_scheduler = build_torch_lr_scheduler(
            get_lr_scheduler_name(config),
            optimizer=optimizer,
            num_warmup_steps=config['train']['warmup_steps'],
            num_training_steps=config['data']['max_steps'],
        )

    if config['train'].get('resume_path'):
        resume_path = config['train']['resume_path']
        if os.path.exists(resume_path):
            if global_rank == 0:
                logger.info("Resuming training from checkpoint: %s", resume_path)
            if use_deepspeed:
                load_dir, tag, _, _ = _resolve_deepspeed_checkpoint(resume_path)
                load_path, client_state = model_engine.load_checkpoint(load_dir, tag=tag)
                if load_path is None:
                    raise RuntimeError(f"DeepSpeed failed to load checkpoint: {load_dir} tag={tag}")
                start_step = client_state.get('step', 0) if client_state else 0
                if global_rank == 0:
                    logger.info("Loaded DeepSpeed checkpoint from %s tag=%s", load_dir, tag)
            else:
                if os.path.isdir(resume_path):
                    raise ValueError(
                        "resume_path points to a DeepSpeed checkpoint directory, but "
                        "train.deepspeed.enabled is false. Enable DeepSpeed for full resume, "
                        "or use train.pretrained_checkpoint to load weights only."
                    )
                checkpoint = _torch_load_checkpoint(resume_path, map_location=device)
                state_dict = checkpoint['model_state_dict']

                if is_distributed:
                    state_dict = _add_module_prefix(state_dict)
                else:
                    state_dict = _strip_module_prefix(state_dict)

                model.load_state_dict(state_dict)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_step = checkpoint['step']
            if global_rank == 0:
                logger.info("Resumed at step %d", start_step)
        else:
            if global_rank == 0:
                logger.warning("Resume path does not exist; starting from scratch: %s", resume_path)

    ema_checkpoint = None
    ema_cfg = config['train'].get('ema', {}) or {}
    ema_enabled = bool(ema_cfg.get('enabled', False))
    if ema_enabled:
        zero_stage = int(config['train'].get('deepspeed', {}).get('zero_stage', 2))
        if use_deepspeed and zero_stage == 3:
            raise ValueError(
                "EMA single-file saving is not supported with DeepSpeed ZeRO-3. "
                "Use ZeRO-0/1/2 or disable train.ema.enabled."
            )
        if global_rank == 0:
            state_dict_getter = (
                (lambda: model_engine.module.state_dict())
                if use_deepspeed
                else (lambda: model.state_dict())
            )
            ema_checkpoint = EMACheckpoint(
                ema_weight=ema_cfg.get('ema_weight', 0.999),
                save_interval=ema_cfg.get('save_interval', config['project']['save_interval']),
                save_dir=save_dir,
                config=config,
                state_dict_getter=state_dict_getter,
                logger=logger,
            )
            ema_checkpoint.load_or_initialize(start_step)
            logger.info(
                "EMA enabled: ema_weight=%.6f save_interval=%d latest=%s final=%s",
                ema_checkpoint.ema_weight,
                ema_checkpoint.save_interval,
                ema_checkpoint.latest_path,
                ema_checkpoint.final_path,
            )

    train_model = model_engine if use_deepspeed else model
    step = _run_training_loop(
        train_model=train_model,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        save_dir=save_dir,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_distributed=is_distributed,
        use_deepspeed=use_deepspeed,
        model_engine=model_engine if use_deepspeed else None,
        global_rank=global_rank,
        max_steps=config['data']['max_steps'],
        start_step=start_step,
        stage_desc="Finetuning",
        log_prefix="train",
        ds_save_prefix="ds_checkpoint",
        file_save_prefix="checkpoint",
        enable_save=True,
        logger=logger,
        ema_checkpoint=ema_checkpoint,
    )

    if global_rank == 0:
        logger.info("Finetuning finished")
    if use_deepspeed:
        ds_save_dir = os.path.join(save_dir, "ds_checkpoint_final")
        model_engine.save_checkpoint(ds_save_dir, client_state={'step': step, 'config': config})
        if global_rank == 0:
            logger.info("Saved final DeepSpeed checkpoint to %s", ds_save_dir)
    else:
        if global_rank == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'config': config
            }, os.path.join(save_dir, "checkpoint_final.pt"))

    if ema_checkpoint is not None:
        ema_checkpoint.finalize(step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_train_config.yaml", help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed launcher (also set via LOCAL_RANK env var)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
