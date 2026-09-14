from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, SiglipImageProcessor
from src.models.VLANeXt import VLANeXt, LlamaProcessorWrapper
from src.datasets.language_action import format_language_action_prompt, normalize_language_action_format

def _get_checkpoint_path(cfg) -> str:
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "finetuned_checkpoint"):
        return cfg.eval.finetuned_checkpoint
    raise ValueError("cfg.eval.finetuned_checkpoint is required")


def _load_checkpoint(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if path.is_file():
        ckpt = torch.load(str(path), map_location="cpu")
        ckpt["trained_with_deepspeed"] = False  # single-file checkpoints are assumed to be non-DeepSpeed
        return ckpt

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    latest_file = path / "latest"
    if latest_file.exists():
        tag = latest_file.read_text().strip()
    else:
        subdirs = sorted(d for d in path.iterdir() if d.is_dir())
        if not subdirs:
            raise FileNotFoundError(
                f"DeepSpeed checkpoint dir {path} has no `latest` file and no subdirs"
            )
        tag = subdirs[-1].name

    tag_dir = path / tag
    mp_rank_file = tag_dir / "mp_rank_00_model_states.pt"
    if not mp_rank_file.exists():
        raise FileNotFoundError(
            f"Expected `{mp_rank_file}` in DeepSpeed checkpoint"
        )

    client_state = torch.load(str(mp_rank_file), map_location="cpu", weights_only=False)
    config = client_state.get("config")
    if config is None:
        raise ValueError(
            f"DeepSpeed client_state at {mp_rank_file} has no `config` key; "
            "train.py must save it via `client_state={'step': step, 'config': config}`"
        )
    step = client_state.get("step", 0)

    print(f"Consolidating ZeRO shards from {path} (tag={tag})")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(path), tag=tag)

    return {
        "config": config,
        "model_state_dict": state_dict,
        "step": step,
        "trained_with_deepspeed": True,
    }


def get_vla(cfg):
    checkpoint_path = _get_checkpoint_path(cfg)
    print(f"Loading model from {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    train_config = checkpoint['config']
    model_config = train_config['model']
    data_config = train_config['data']
    
    ckpt_backbone_mode = model_config.get('backbone_mode', 'finetune')
    print(f"Backbone mode: {ckpt_backbone_mode}")

    attn_implementation = "flash_attention_2"

    num_train_timesteps = model_config.get('num_train_timesteps', model_config.get('diffusion_steps', 1000))
    ckpt_inf_steps = model_config.get('num_inference_timesteps', model_config.get('diffusion_steps', 10))
    
    num_inference_timesteps = ckpt_inf_steps
    if hasattr(cfg, "model") and hasattr(cfg.model, "diffusion_steps"):
        num_inference_timesteps = cfg.model.diffusion_steps
        print(f"Overriding inference diffusion steps to: {num_inference_timesteps}")

    scheduler_type = model_config['scheduler_type']
    if hasattr(cfg, "model"):
        scheduler_type = getattr(cfg.model, "scheduler_type", scheduler_type)

    use_action_input_policy = model_config.get('use_action_input_policy', False)
    use_transformer_proprio_projector = model_config.get('use_transformer_proprio_projector', False)

    model = VLANeXt(
        lmm_path=model_config['lmm_path'],
        vision_encoder_path=model_config.get('vision_encoder_path', "google/siglip2-base-patch16-256"),
        use_pretrained_backbone=model_config.get('use_pretrained_backbone', True),
        action_dim=model_config['action_dim'],
        num_actions=data_config['future_len'],
        num_queries=model_config['num_queries'],
        num_history=data_config['history_len'],
        loss_type=model_config.get('loss_type', 'diffusion'),
        future_image_loss_weight=float(model_config.get('future_image_loss_weight', 0.0)),
        num_train_timesteps=num_train_timesteps,
        num_inference_timesteps=num_inference_timesteps,
        scheduler_type=scheduler_type,
        diffusion_loss_domain=model_config.get('diffusion_loss_domain', 'noise'),
        condition_type=model_config.get('condition_type', 'loose'),
        policy_hidden_size=model_config.get('policy_hidden_size', 1024),
        policy_depth=model_config.get('policy_depth', 29),
        policy_num_heads=model_config.get('policy_num_heads', 12),
        policy_mlp_ratio=model_config.get('policy_mlp_ratio', 4.0),
        policy_pos_embed=model_config.get('policy_pos_embed', 'absolute'),
        use_proprio_input_vlm=model_config.get('use_proprio_input_vlm', True),
        use_action_input_policy=use_action_input_policy,
        use_transformer_proprio_projector=use_transformer_proprio_projector,
        projector_depth=model_config['projector_depth'],
        projector_num_heads=model_config['projector_num_heads'],
        use_transformer_connector=model_config['use_transformer_connector'],
        connector_depth=model_config['connector_depth'],
        connector_num_heads=model_config['connector_num_heads'],
        backbone_mode=ckpt_backbone_mode,
        gradient_checkpointing=False,
        num_bins=model_config.get('num_bins', 256),
        classification_type=model_config.get('classification_type', 'parallel'),
        fast_action_tokenizer=model_config.get('fast_action_tokenizer', None),
        generator_hidden_size=model_config.get('generator_hidden_size', 768),
        generator_depth=model_config.get('generator_depth', model_config.get('policy_depth', 12)),
        generator_num_heads=model_config.get('generator_num_heads', 12),
        generator_mlp_ratio=model_config.get('generator_mlp_ratio', 4.0),
        generator_max_seq_len=model_config.get('generator_max_seq_len', 1024),
        future_image_num_tokens=model_config.get('future_image_num_tokens', 256),
        future_image_prediction_type=model_config.get('future_image_prediction_type', 'emu_token'),
        future_image_dino_model_path=model_config.get(
            'future_image_dino_model_path',
            'facebook/dinov3-vitb16-pretrain-lvd1689m',
        ),
        future_image_dino_image_size=model_config.get('future_image_dino_image_size', 256),
        future_image_flow_num_inference_timesteps=model_config.get(
            'future_image_flow_num_inference_timesteps',
            10,
        ),
        attn_implementation=attn_implementation,
        language_action_loss_weight=model_config.get('language_action_loss_weight', 0.0),
        policy_gradient_stop_for_vlm=model_config.get('policy_gradient_stop_for_vlm', False),
        dct_loss_weight=model_config.get('dct_loss_weight', 0.1),
        dct_low_freq_weight=model_config.get('dct_low_freq_weight', 1.0),
        dct_high_freq_weight=model_config.get('dct_high_freq_weight', 1.0),
        dct_freq_split=model_config.get('dct_freq_split', 0.125),
        dct_similarity_type=model_config.get('dct_similarity_type', 'mae'),
        video_generation_loss_weight=model_config.get('video_generation_loss_weight', 1.0),
        wan_action_condition_mode=model_config.get('wan_action_condition_mode', 'fast'),
        wan_flow_shift=model_config.get('wan_flow_shift', 5.0),
        wan_text_len=model_config.get('wan_text_len', 512),
        wan_tokenizer_model_id=model_config.get('wan_tokenizer_model_id', 'google/umt5-xxl'),
        future_video_downsample=data_config.get('future_video_downsample', model_config.get('future_video_downsample', 1)),
    )
    
    
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded state dict. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trained_with_deepspeed = checkpoint.get("trained_with_deepspeed", False)
    if trained_with_deepspeed:
        model.to(dtype=torch.bfloat16)
        print("Checkpoint trained with DeepSpeed bf16; casting model to bfloat16 for eval.")
    else:
        print("Checkpoint trained in fp32/autocast; keeping model in fp32 and using autocast at eval.")

    model.eval()

    model.train_config = train_config
    model.trained_with_deepspeed = trained_with_deepspeed

    return model

def get_processor(cfg):
    checkpoint_path = _get_checkpoint_path(cfg)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    lmm_path = config["model"]["lmm_path"]

    if "wan" in lmm_path.lower():
        return None

    if "llama" in lmm_path.lower():
        vision_encoder_path = config["model"].get("vision_encoder_path", "google/siglip2-base-patch16-256")
        tokenizer = AutoTokenizer.from_pretrained(lmm_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
        return LlamaProcessorWrapper(tokenizer, image_processor)

    return AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)

def get_vla_action(cfg, model, processor, obs, task_label):
    data_cfg = model.train_config["data"]
    model_cfg = model.train_config["model"]
    input_modality = data_cfg.get("input_modality", "image")
    view_mode = data_cfg.get("view_mode", "single")
    fps = float(data_cfg.get("fps", 20.0))
    history_len = getattr(model, "num_history", 0)
    device = next(model.parameters()).device
    effective_processor = getattr(model, "processor", processor)
    is_batched = isinstance(obs, (list, tuple))
    obs_batch = list(obs) if is_batched else [obs]
    task_labels = list(task_label) if isinstance(task_label, (list, tuple)) else [task_label] * len(obs_batch)
    
    processor_name = effective_processor.__class__.__name__ if effective_processor is not None else ""
    is_wan = getattr(model, "model_family", None) == "wan"
    is_paligemma = "PaliGemma" in processor_name
    is_qwen = "Qwen" in processor_name
    is_llama = "Llama" in processor_name
    generate_language_action = bool(getattr(model, "enable_language_action_loss", False))
    language_action_format = normalize_language_action_format(
        model_cfg.get("language_action_format", "lap")
    )
    language_action_max_new_tokens = int(model_cfg.get("language_action_max_new_tokens", 256))

    def _policy_prompt(label):
        if not generate_language_action:
            return label
        frame_description = "normalized action space" if language_action_format == "vla-0" else "robot base frame"
        return (
            format_language_action_prompt(
                label,
                language_action_format=language_action_format,
                frame_description=frame_description,
                future_len=data_cfg.get("future_len", getattr(model, "num_actions", 1)),
                action_dim=getattr(model, "action_dim", 7),
            )
            + "\nAnswer: "
        )

    prompt_labels = [_policy_prompt(label) for label in task_labels]

    def _take_last(history_list, fallback, n: int):
        if not history_list:
            history_list = [fallback]
        if n > 0:
            xs = history_list[-n:]
            if len(xs) < n:
                xs = ([xs[0]] * (n - len(xs))) + xs
            return xs
        return [fallback]

    def _obs_media(sample):
        all_images = sample.get("image_history", [sample["full_image"]])
        images_np = _take_last(all_images, sample["full_image"], history_len)
        pil_images = [Image.fromarray(img) for img in images_np]

        media = {"main": pil_images, "wrist": None, "wrist2": None}
        has_wrist2 = view_mode == "multi" and (
            sample.get("image_history_wrist2") or sample.get("full_image_wrist2") is not None
        )

        if view_mode == "multi":
            all_wrist = sample.get("image_history_wrist", [sample["full_image_wrist"]])
            wrist_np = _take_last(all_wrist, sample["full_image_wrist"], history_len)
            media["wrist"] = [Image.fromarray(img) for img in wrist_np]
            if has_wrist2:
                fallback_wrist2 = sample.get("full_image_wrist2") or sample["image_history_wrist2"][0]
                all_wrist2 = sample.get("image_history_wrist2", [fallback_wrist2])
                wrist2_np = _take_last(all_wrist2, fallback_wrist2, history_len)
                media["wrist2"] = [Image.fromarray(img) for img in wrist2_np]
        return media

    media_batch = [_obs_media(sample) for sample in obs_batch]

    proprioception = None
    if model.use_proprio_input_vlm:
        proprio_batch = []
        for sample in obs_batch:
            all_states = sample.get("state_history", [])
            if history_len > 0:
                states = all_states[-history_len:]
                if len(states) < history_len:
                    states = ([states[0]] * (history_len - len(states))) + states if states else [np.zeros(model.action_dim)] * history_len
            else:
                states = []
            if len(states) > 0:
                proprio_batch.append(np.stack(states))
        if len(proprio_batch) > 0:
            proprioception = torch.tensor(np.stack(proprio_batch), dtype=torch.bfloat16).to(device)

    history_actions = None
    if getattr(model, 'use_action_input_policy', False):
        action_batch = []
        for sample in obs_batch:
            all_actions = sample.get("action_history", [])
            if history_len > 0:
                actions = all_actions[-history_len:]
                if len(actions) < history_len:
                    if "history_action_pad" not in sample:
                        raise ValueError("history_action_pad is required to pad LIBERO action history.")
                    pad_action = np.asarray(sample["history_action_pad"], dtype=np.float32)
                    actions = ([pad_action] * (history_len - len(actions))) + actions
            else:
                actions = []
            if len(actions) > 0:
                action_batch.append(np.stack(actions))
        if len(action_batch) > 0:
            history_actions = torch.tensor(np.stack(action_batch), dtype=torch.bfloat16).to(device)

    trained_with_deepspeed = getattr(model, "trained_with_deepspeed", False)
    if trained_with_deepspeed:
        autocast_ctx = torch.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if is_wan:
        wan_prompt_template = model_cfg.get(
            "wan_prompt_template",
            "A video recorded from a robot's point of view executing the instruction: {instruction}",
        )
        current_images = []
        for media in media_batch:
            pieces = [np.asarray(media["main"][-1], dtype=np.uint8)]
            if view_mode == "multi":
                pieces.append(np.asarray(media["wrist"][-1], dtype=np.uint8))
            image_np = np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
            image = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
            current_images.append(image)
        current_images = torch.stack(current_images).to(device=device, dtype=torch.bfloat16)
        prompt_texts = [wan_prompt_template.format(instruction=label, task=label) for label in task_labels]
        with torch.no_grad(), autocast_ctx:
            action_pred = model.predict_action(
                prompt_texts=prompt_texts,
                current_images=current_images,
                proprioception=proprioception,
                history_actions=history_actions,
            )
        action_np = action_pred.float().cpu().numpy()
        return action_np if is_batched else action_np[0]

    inputs = {}
    
    if input_modality == "video":
        if is_paligemma or is_llama:
            raise ValueError(f"{effective_processor.__class__.__name__} implementation in VLANeXt currently supports 'image' modality only.")
        texts = []
        videos = []
        video_metadata = []
        for media, label in zip(media_batch, prompt_labels):
            content = []
            sample_videos = [media["main"]]
            if view_mode == "multi":
                sample_videos.append(media["wrist"])
                if media["wrist2"] is not None:
                    sample_videos.append(media["wrist2"])

            for video in sample_videos:
                content.append({"type": "video", "video": video})
                videos.append(video)
                video_metadata.append({
                    "total_num_frames": len(video),
                    "fps": fps,
                    "frames_indices": list(range(len(video))),
                })
            content.append({"type": "text", "text": label})
            messages = [{"role": "user", "content": content}]
            texts.append(effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = effective_processor(
            text=texts,
            videos=videos,
            videos_kwargs={"fps": fps, "return_metadata": True, "video_metadata": video_metadata},
            padding=True,
            return_tensors="pt",
        )

    elif input_modality == "image":
        images = []
        sample_image_counts = []
        for media in media_batch:
            sample_images = [media["main"][-1]]
            if view_mode == "multi":
                sample_images.append(media["wrist"][-1])
                if media["wrist2"] is not None:
                    sample_images.append(media["wrist2"][-1])
            images.extend(sample_images)
            sample_image_counts.append(len(sample_images))

        if is_llama:
            inputs = effective_processor.tokenizer(prompt_labels, padding=True, return_tensors="pt")
            image_inputs = effective_processor.image_processor(images, return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]

        elif is_paligemma:
            text = [
                "<image>" * num_images + label
                for num_images, label in zip(sample_image_counts, prompt_labels)
            ]
            inputs = effective_processor(
                text=text,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_qwen:
            texts = []
            image_offset = 0
            for num_images, label in zip(sample_image_counts, prompt_labels):
                content = []
                for img in images[image_offset:image_offset + num_images]:
                    content.append({"type": "image", "image": img})
                image_offset += num_images
                content.append({"type": "text", "text": label})
                messages = [{"role": "user", "content": content}]
                texts.append(effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            inputs = effective_processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
    else:
        raise ValueError(f"Unknown input_modality: {input_modality} for model type")

    valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids", "token_type_ids", "prompt_texts", "current_images"}
    inputs = {k: v.to(device) for k, v in inputs.items() if k in valid_keys}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)

    with torch.no_grad(), autocast_ctx:
        action_pred = model.predict_action(
            proprioception=proprioception,
            history_actions=history_actions,
            generate_language_action=generate_language_action,
            language_action_max_new_tokens=language_action_max_new_tokens,
            **inputs,
        )

    action_np = action_pred.float().cpu().numpy()
    return action_np if is_batched else action_np[0]
