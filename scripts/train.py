import os
import yaml
import argparse
import wandb
import random
import re
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
from src.models.rt2_like_baseline import RT2LikeBaseline
from src.datasets.libero_act import LiberoAct
from src.datasets.droid_act import DroidAct


# -----------------------------------------------------------------------------
# -------------------------------- Prequisite ---------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        load_future_image=False,
    ):
        self.processor = processor
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.fps = float(fps)
        self.load_future_image = load_future_image

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

    def _to_pil(self, img_np: np.ndarray) -> Image.Image:
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _uniform(self, a: float, b: float) -> float:
        return float(np.random.uniform(a, b))

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

    def __call__(self, batch):
        texts = []
        fps = self.fps

        images = []
        videos = []
        
        gt_actions_list = []
        proprio_list = []
        hist_actions_list = []
        future_images_list = []

        is_paligemma = "PaliGemma" in self.processor.__class__.__name__
        is_qwen = "Qwen" in self.processor.__class__.__name__
        is_llama = "Llama" in self.processor.__class__.__name__

        for sample in batch:
            instruction = sample["instruction"]

            if is_paligemma:
                im0 = self._augment_frames_uint8(sample["image"])
                num_imgs = 1
                if self.view_mode == "multi":
                    im1 = self._augment_frames_uint8(sample["image_wrist"])
                    images.extend([im0, im1])
                    num_imgs = 2
                else:
                    images.append(im0)
                
                texts.append("<image>" * num_imgs + instruction)

            elif is_llama:
                im0 = self._augment_frames_uint8(sample["image"])
                if self.view_mode == "multi":
                    im1 = self._augment_frames_uint8(sample["image_wrist"])
                    images.extend([im0, im1])
                else:
                    images.append(im0)
                
                texts.append(instruction)

            elif is_qwen:
                content = []
                if self.input_modality == "video":
                    v0 = self._augment_frames_uint8(sample["video"])
                    if self.view_mode == "multi":
                        v1 = self._augment_frames_uint8(sample["video_wrist"])
                        content.extend([{"type": "video", "video": v0}, {"type": "video", "video": v1}])
                        videos.extend([v0, v1]) 
                    else:
                        content.append({"type": "video", "video": v0})
                        videos.append(v0)

                elif self.input_modality == "image":
                    im0 = self._augment_frames_uint8(sample["image"])
                    if self.view_mode == "multi":
                        im1 = self._augment_frames_uint8(sample["image_wrist"])
                        content.extend([{"type": "image", "image": im0}, {"type": "image", "image": im1}])
                        images.extend([im0, im1])
                    else:
                        content.append({"type": "image", "image": im0})
                        images.append(im0)
                else:
                    raise ValueError(f"Unknown input_modality: {self.input_modality}")

                content.append({"type": "text", "text": instruction})

                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)

            gt_actions_list.append(sample["future_actions"])
            proprio_list.append(sample["proprioception"])
            hist_actions_list.append(sample["history_actions"])
            
            if self.load_future_image and "future_image" in sample:
                f_img = sample["future_image"]
                f_img = torch.from_numpy(f_img).permute(2, 0, 1).float() / 127.5 - 1.0
                future_images_list.append(f_img)

        if is_paligemma:
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_llama:
            inputs = self.processor.tokenizer(
                texts,
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
                    text=texts,
                    videos=videos,
                    videos_kwargs={"fps": fps, "return_metadata": True, "video_metadata": video_metadata}, 
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )

        gt_actions = torch.stack(gt_actions_list)
        proprio = torch.stack(proprio_list) if self.use_proprio_input_vlm else None
        hist_actions = torch.stack(hist_actions_list) if self.use_action_input_policy else None
        future_images = torch.stack(future_images_list) if self.load_future_image else None
        
        return inputs, gt_actions, proprio, hist_actions, future_images

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    # -----------------------------------------------------------------------------
    # ----------------------------------- Setup -----------------------------------
    # -----------------------------------------------------------------------------
    is_distributed = config['train'].get('distributed', False)
    gradient_accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
    use_proprio_input_vlm = config['model'].get('use_proprio_input_vlm', True)
    use_action_input_policy = config['model'].get('use_action_input_policy', True)
    future_image_loss_weight = float(config['model'].get('future_image_loss_weight', 0.0))
    enable_future_image_loss = (future_image_loss_weight > 0)
    load_future_image = enable_future_image_loss
    future_image_mode = config['model'].get('future_image_mode', 'horizon')
    input_modality = config["data"].get("input_modality", "video")
    view_mode = config["data"].get("view_mode", "single")
    augmentation = config["data"].get("augmentation", {})
    dataset_name = config['data'].get('dataset_name', 'libero')
    if dataset_name == "droid":
        fps = 15.0
    elif dataset_name == "libero":
        fps = 20.0
    else:
        fps = 20.0
    full_sequence = bool(config['data'].get('full_sequence', False))
    seed = config['train'].get('seed', 42)

    set_seed(seed)

    if is_distributed:
        dist.init_process_group(backend=config['train']['dist_backend'])
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"[Rank {global_rank}] Initialized process group")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device(config['train'].get('device', 'cuda'))
        
    os.makedirs(config['project']['output_dir'], exist_ok=True)
    save_dir = ""
    wandb_project = config['project'].get('wandb_project', 'VLANeXt')
    wandb_name = config['project']['name']
    if global_rank == 0:
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
        os.makedirs(save_dir, exist_ok=True)
    if config['project'].get('use_wandb', False) and global_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=config['project'].get('wandb_entity', None),
            name=wandb_name,
            config=config
        )

    pretrained_ckpt_path = config['train'].get('pretrained_checkpoint')
    has_pretrained_ckpt = (pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path))
    model_type = config['model'].get('model_type', 'vlanext')
    if model_type == 'rt2_baseline':
        if global_rank == 0:
            print("Initializing RT2LikeBaseline model...")
        model = RT2LikeBaseline(
            lmm_path=config['model']['lmm_path'],
            vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=config['model']['action_dim'],
            num_actions=config['data']['future_len'],
            num_history=config['data']['history_len'],
            use_proprio_input_vlm=use_proprio_input_vlm,
            use_transformer_projector=config['model'].get('use_transformer_proprio_projector', True),
            projector_depth=config['model']['projector_depth'],
            projector_num_heads=config['model']['projector_num_heads'],
            backbone_mode=config['model'].get('backbone_mode', 'finetune'),
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
            num_bins=config['model'].get('num_bins', 256),
        ).to(device, dtype=torch.bfloat16)
    else:
        model = VLANeXt(
            lmm_path=config['model']['lmm_path'],
            vision_encoder_path=config['model'].get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=config['model']['action_dim'],
            num_actions=config['data']['future_len'],
            num_queries=config['model']['num_queries'],
            num_history=config['data']['history_len'],
            loss_type=config['model'].get('loss_type', 'diffusion'),
            future_image_loss_weight=future_image_loss_weight,
            num_train_timesteps=config['model'].get('num_train_timesteps', 1000),
            num_inference_timesteps=config['model'].get('num_inference_timesteps', 10),
            scheduler_type=config['model']['scheduler_type'],
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
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False),
            num_bins=config['model'].get('num_bins', 256),
            generator_hidden_size=config['model'].get('generator_hidden_size', 768),
            generator_depth=config['model'].get('generator_depth', 12),
            generator_num_heads=config['model'].get('generator_num_heads', 12),
            generator_mlp_ratio=config['model'].get('generator_mlp_ratio', 4.0),
            action_vqvae=config['model'].get('action_vqvae', None),

            dct_loss_weight=config['model'].get('dct_loss_weight', 0.1),
            dct_low_freq_weight=config['model'].get('dct_low_freq_weight', 1.0),
            dct_high_freq_weight=config['model'].get('dct_high_freq_weight', 1.0),
            dct_freq_split=config['model'].get('dct_freq_split', 0.125),
            dct_similarity_type=config['model'].get('dct_similarity_type', 'mae'),
        ).to(device, dtype=torch.bfloat16)
    if has_pretrained_ckpt:
        if global_rank == 0:
            print(f"Loading pretrained VLA checkpoint: {pretrained_ckpt_path}")
        checkpoint = torch.load(pretrained_ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if global_rank == 0:
            print(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        if global_rank == 0:
            if pretrained_ckpt_path:
                print(f"Warning: Pretrained checkpoint path '{pretrained_ckpt_path}' does not exist. Training from scratch.")
            else:
                print("No pretrained checkpoint provided. Training from scratch.")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        model_unwrapped = model.module
    else:
        model_unwrapped = model

    # -----------------------------------------------------------------------------
    # -------------------------------- Data Loader --------------------------------
    # -----------------------------------------------------------------------------
    data_root = config['data']['data_root']
    if dataset_name == "droid":
        droid_path = os.path.join(data_root, "1.0.1")
        if global_rank == 0:
            print(f"Initializing DROID Dataset: {droid_path}")
    else:
        task_suite = config['data']['task_suite_name']
        libero_path = os.path.join(data_root, task_suite, "1.0.0")
        if global_rank == 0:
            print(f"Initializing Libero Dataset: {libero_path}")
    collator = DataCollatorForVLANeXt(
        processor=model_unwrapped.processor,
        use_proprio_input_vlm=use_proprio_input_vlm,
        use_action_input_policy=use_action_input_policy,
        input_modality=input_modality,
        view_mode=view_mode,
        fps=fps,
        augmentation=augmentation,
        load_future_image=load_future_image,
    )
    total_batch_size = config['data']['batch_size']
    per_device_batch_size = total_batch_size // (world_size * gradient_accumulation_steps)
    if global_rank == 0:
        print(f"Total Batch Size: {total_batch_size}, World Size: {world_size}, Grad Acc Steps: {gradient_accumulation_steps}, Per-Device Batch Size: {per_device_batch_size}")

    default_buffer_size = 30000 if dataset_name == "droid" else 10000
    buffer_size = config['data'].get("buffer_size", default_buffer_size)
    if global_rank == 0:
        print(f"Dataset Shuffle Buffer Size: {buffer_size}")

    def create_dataloader(history_len):
        if dataset_name == "droid":
            ds = DroidAct(
                droid_path=droid_path,
                dataset_name="droid",
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
            )
        else:
            ds = LiberoAct(
                data_path=libero_path,
                dataset_name=task_suite,
                history_len=history_len,
                future_len=config['data']['future_len'],
                full_sequence=full_sequence,
                input_modality=input_modality,
                view_mode=view_mode,
                load_future_image=load_future_image,
                future_image_mode=future_image_mode,
                buffer_size=buffer_size,
            )
        return DataLoader(
            ds, 
            batch_size=per_device_batch_size,
            num_workers=config['data']['num_workers'],
            collate_fn=collator
        )

    dataloader = create_dataloader(config['data']['history_len'])

    # -----------------------------------------------------------------------------
    # ---------------------- VQ-VAE Training (if applicable) ----------------------
    # -----------------------------------------------------------------------------
    vqvae_config = config['model'].get('action_vqvae', {})
    if (
        vqvae_config.get('enabled', False) 
        and not config['train'].get('resume_path')
    ):
        if global_rank == 0:
            print("\n=== Starting Action VQ-VAE Pre-training ===")
        vqvae_params = list(model_unwrapped.action_vqvae.parameters())
        vqvae_optim = AdamW(
            vqvae_params,
            lr=float(vqvae_config.get('learning_rate', 1e-3)),
            weight_decay=float(config['train']['weight_decay'])
        )
        vqvae_steps = vqvae_config.get('steps', 1000)
        vqvae_pbar = tqdm(total=vqvae_steps, desc="Pre-training VQ-VAE", disable=global_rank != 0)
        vqvae_iter = iter(dataloader)
        model.train()
        
        for i in range(vqvae_steps):
            try:
                batch = next(vqvae_iter)
            except StopIteration:
                vqvae_iter = iter(dataloader)
                batch = next(vqvae_iter)
            _, gt_actions, _, _, _ = batch
            gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
            vqvae_optim.zero_grad()
            loss = model(actions=gt_actions, task="action_vqvae_pretrain")
            loss.backward()
            vqvae_optim.step()
            if global_rank == 0:
                vqvae_pbar.update(1)
                vqvae_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if config['project'].get('use_wandb', False) and i % config['project']['log_interval'] == 0:
                    wandb.log({"vqvae_pretrain/loss": loss.item(), "vqvae_pretrain/step": i})
        
        if vqvae_config.get('frozen', True):
            if global_rank == 0: print("Freezing VQ-VAE parameters after pretraining.")
            model_unwrapped.action_vqvae.requires_grad_(False)
            model_unwrapped.action_vqvae.eval()
        else:
            if global_rank == 0: print("Keeping VQ-VAE parameters trainable (finetuning).")
            model_unwrapped.action_vqvae.requires_grad_(True)
            model_unwrapped.action_vqvae.train()
            
        if global_rank == 0:
            vqvae_pbar.close()
            print("=== Action VQ-VAE Pre-training Finished ===\n")
            torch.save(model_unwrapped.action_vqvae.state_dict(), os.path.join(save_dir, "action_vqvae_pretrained.pt"))

    # -----------------------------------------------------------------------------
    # ------------------------ Action Generation Training -------------------------
    # -----------------------------------------------------------------------------
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config['train']['learning_rate']),
        weight_decay=float(config['train']['weight_decay'])
    )
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config['train']['warmup_steps'],
        num_training_steps=config['data']['max_steps']
    )

    start_step = 0
    if config['train'].get('resume_path'):
        resume_path = config['train']['resume_path']
        if os.path.exists(resume_path):
            if global_rank == 0:
                print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            state_dict = checkpoint['model_state_dict']

            if is_distributed and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not is_distributed and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            if global_rank == 0:
                print(f"Resumed at step {start_step}")
        else:
            if global_rank == 0:
                print(f"Warning: Resume path {resume_path} does not exist. Starting from scratch.")
    
    model.train()
    step = start_step
    batch_idx = 0
    if global_rank == 0:
        progress_bar = tqdm(total=config['data']['max_steps'], initial=start_step, desc="Finetuning")
    data_iter = iter(dataloader)
    optimizer.zero_grad()
    while step < config['data']['max_steps']:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        inputs, gt_actions, proprio, hist_actions, future_images = batch
        del batch
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        del inputs
        for k in ['pixel_values', 'pixel_values_videos']:
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].to(dtype=torch.bfloat16)

        gt_actions = gt_actions.to(device, dtype=torch.bfloat16)
        if proprio is not None:
            proprio = proprio.to(device, dtype=torch.bfloat16)
        if hist_actions is not None:
            hist_actions = hist_actions.to(device, dtype=torch.bfloat16)
        if future_images is not None:
            future_images = future_images.to(device, dtype=torch.bfloat16)
        
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "token_type_ids"}
        forward_args = {k: v for k, v in model_inputs.items() if k in valid_keys}
        do_update = (batch_idx + 1) % gradient_accumulation_steps == 0
        sync_context = model.no_sync if (is_distributed and not do_update) else nullcontext
        with sync_context():
            loss = model(
                actions=gt_actions,
                proprioception=proprio,
                history_actions=hist_actions,
                future_images=future_images,
                **forward_args
            )
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        if do_update:
            if config['train']['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step += 1
            if global_rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            if step % config['project']['log_interval'] == 0 and global_rank == 0:
                if config['project'].get('use_wandb', False):
                    wandb.log({
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "step": step
                    })
            if step % config['project']['save_interval'] == 0 and global_rank == 0:
                save_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'config': config
                }, save_path)
                print(f"\nSaved checkpoint to {save_path}")
                
        batch_idx += 1

    if global_rank == 0:
        print("Finetuning finished.")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'config': config
        }, os.path.join(save_dir, "checkpoint_final.pt"))

        if config['project'].get('use_wandb', False):
            wandb.finish()
            
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
