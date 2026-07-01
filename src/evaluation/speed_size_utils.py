import time
import numpy as np
from PIL import Image

import torch


def get_model_info(model):
    param_size = 0
    param_count = 0
    seen = set()
    for param in model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            param_count += param.nelement()
            param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        if id(buffer) not in seen:
            seen.add(id(buffer))
            buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return param_count, size_mb


def _build_dummy_inputs(processor, input_modality, view_mode, bimanual, fps, batch_size, device, dtype):
    is_paligemma = "PaliGemma" in processor.__class__.__name__
    is_llama = "Llama" in processor.__class__.__name__
    is_qwen = "Qwen" in processor.__class__.__name__

    text = "Pick up the red block and place it on the blue plate."

    if input_modality == "image":
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        imgs_per_sample = (3 if bimanual else 2) if view_mode == "multi" else 1

        if is_paligemma:
            text_inputs = ["<image>" * imgs_per_sample + text] * batch_size
            images = [img] * (batch_size * imgs_per_sample)
            inputs = processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
        elif is_llama:
            text_inputs = [text] * batch_size
            inputs = processor.tokenizer(text_inputs, padding=True, return_tensors="pt")
            image_inputs = processor.image_processor([img] * (batch_size * imgs_per_sample), return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]
        elif is_qwen:
            messages = []
            for _ in range(batch_size):
                content = [{"type": "image", "image": img}] * imgs_per_sample
                content.append({"type": "text", "text": text})
                messages.append([{"role": "user", "content": content}])
            text_inputs = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            images = [img] * (batch_size * imgs_per_sample)
            inputs = processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported processor: {processor.__class__.__name__}")

    elif input_modality == "video":
        if not is_qwen:
            raise NotImplementedError("Video modality is only supported for Qwen-based processors.")
        num_frames = 8
        frames = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(num_frames)]
        vids_per_sample = (3 if bimanual else 2) if view_mode == "multi" else 1

        all_videos = [frames] * (batch_size * vids_per_sample)
        video_metadata = [
            {"total_num_frames": num_frames, "fps": fps, "frames_indices": list(range(num_frames))}
            for _ in all_videos
        ]
        messages = []
        for _ in range(batch_size):
            content = [{"type": "video", "video": frames}] * vids_per_sample
            content.append({"type": "text", "text": text})
            messages.append([{"role": "user", "content": content}])
        text_inputs = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = processor(
            text=text_inputs,
            videos=all_videos,
            videos_kwargs={"fps": fps, "return_metadata": True, "video_metadata": video_metadata},
            padding=True,
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown input_modality: {input_modality}")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    for k in ["pixel_values", "pixel_values_videos"]:
        if k in inputs:
            inputs[k] = inputs[k].to(dtype)
    return inputs


def run_size_speed_eval(model, config, device, dtype, input_modality, view_mode, bimanual, fps):
    """Run model size and inference speed evaluation based on config settings.

    Called during training (rank 0 only) after model initialization.
    Controlled by the `size_speed_eval` section in the config.

    Returns:
        bool: True if training should exit after evaluation, False otherwise.
    """
    eval_cfg = config.get("size_speed_eval", {})
    if not eval_cfg.get("enabled", False):
        return False

    batch_size = eval_cfg.get("batch_size", 1)
    num_warmup = eval_cfg.get("num_warmup", 5)
    num_runs = eval_cfg.get("num_runs", 50)

    print("\n" + "=" * 60)
    print("       Model Size & Inference Speed Evaluation")
    print("=" * 60)

    # Size
    param_count, size_mb = get_model_info(model)
    print(f"[Size]  Parameters : {param_count / 1e6:.2f} M")
    print(f"[Size]  Memory     : {size_mb:.2f} MB")

    # Speed
    processor = model.processor
    print(f"[Speed] Building dummy inputs (modality={input_modality}, view={view_mode}, bs={batch_size})...")
    try:
        dummy_inputs = _build_dummy_inputs(
            processor, input_modality, view_mode, bimanual, fps, batch_size, device, dtype
        )
    except NotImplementedError as e:
        print(f"[Speed] Skipped: {e}")
        print("=" * 60 + "\n")
        return eval_cfg.get("exit_after_eval", False)

    history_len = getattr(model, "num_history", 1) or 1
    action_dim = getattr(model, "action_dim", 7)
    proprioception = None
    if getattr(model, "use_proprio_input_vlm", False):
        proprioception = torch.randn(batch_size, history_len, action_dim, device=device, dtype=dtype)
    history_actions = None
    if getattr(model, "use_action_input_policy", False):
        history_actions = torch.randn(batch_size, history_len, action_dim, device=device, dtype=dtype)

    valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"}
    fwd_inputs = {k: v for k, v in dummy_inputs.items() if k in valid_keys}

    autocast_ctx = torch.autocast("cuda", dtype=dtype) if torch.cuda.is_available() else torch.autocast("cpu", dtype=dtype)

    was_training = model.training
    model.eval()
    with torch.no_grad(), autocast_ctx:
        print(f"[Speed] Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            model.predict_action(proprioception=proprioception, history_actions=history_actions, **fwd_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"[Speed] Timing ({num_runs} runs)...")
        timings = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                model.predict_action(proprioception=proprioception, history_actions=history_actions, **fwd_inputs)
                t1.record()
                torch.cuda.synchronize()
                timings.append(t0.elapsed_time(t1))
            else:
                s = time.perf_counter()
                model.predict_action(proprioception=proprioception, history_actions=history_actions, **fwd_inputs)
                timings.append((time.perf_counter() - s) * 1000.0)

    if was_training:
        model.train()

    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    throughput = 1000.0 / avg_ms * batch_size
    print(f"[Speed] Latency    : {avg_ms:.2f} ms ± {std_ms:.2f} ms  (bs={batch_size})")
    print(f"[Speed] Throughput : {throughput:.2f} FPS")
    print("=" * 60 + "\n")
    return eval_cfg.get("exit_after_eval", False)
