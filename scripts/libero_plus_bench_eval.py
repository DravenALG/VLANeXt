"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import tqdm
from libero.libero import benchmark
import torch
from pathlib import Path

from src.datasets.libero_act import (
    action_min_spatial, action_max_spatial,
    action_min_object, action_max_object,
    action_min_goal, action_max_goal,
    action_min_10, action_max_10,
)

from src.evaluation.libero_bench.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from src.evaluation.libero_bench.VLANeXt_utils import get_processor as get_vlanext_processor
from src.evaluation.libero_bench.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)


class DictConfig:
    """Helper to convert dict to object for dot notation access."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)


def eval_libero(cfg) -> None:
    assert cfg.eval.finetuned_checkpoint is not None, "cfg.eval.finetuned_checkpoint must not be None!"

    set_seed_everywhere(cfg.eval.seed)

    model = get_model(cfg)
    processor = get_vlanext_processor(cfg)

    checkpoint_path = Path(cfg.eval.finetuned_checkpoint)

    try:
        step = str(checkpoint_path.stem.split('_')[-1])
    except ValueError:
        step = "unknown"

    num_steps_execute = int(getattr(cfg.eval, "num_steps_execute", 1))
    diffusion_steps = int(getattr(cfg.model, "diffusion_steps", -1))

    output_dir = checkpoint_path.parent
    eval_dir = output_dir / (
        f"{cfg.eval.task_suite_name}_checkpoint_{step}"
        f"_exec{num_steps_execute}"
        f"_diff{diffusion_steps}"
        f"_libero_plus"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    local_log_filepath = eval_dir / "log.txt"
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.eval.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.eval.task_suite_name}")
    log_file.write(f"Task suite: {cfg.eval.task_suite_name}\n")

    json_path = Path("/data/NTU_slab/draven/proj/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json")
    task_name_to_category = {}
    if json_path.exists():
        with open(json_path, 'r') as f:
            task_classification = json.load(f)
        suite_tasks = task_classification.get(cfg.eval.task_suite_name, [])
        task_name_to_category = {t['name']: t['category'] for t in suite_tasks}
    else:
        print(f"Warning: Task classification file not found at {json_path}")
    category_name_map = {
        "Background Textures": "Background",
        "Robot Initial States": "Robot",
        "Camera Viewpoints": "Camera",
        "Language Instructions": "Language",
        "Sensor Noise": "Noise",
        "Objects Layout": "Layout",
        "Light Conditions": "Light"
    }
    category_stats = {v: {'success': 0, 'total': 0} for v in category_name_map.values()}

    train_suite_name = model.train_config['data']['task_suite_name']

    if 'spatial' in train_suite_name:
        action_min = np.array(action_min_spatial)
        action_max = np.array(action_max_spatial)
    elif 'object' in train_suite_name:
        action_min = np.array(action_min_object)
        action_max = np.array(action_max_object)
    elif 'goal' in train_suite_name:
        action_min = np.array(action_min_goal)
        action_max = np.array(action_max_goal)
    elif '10' in train_suite_name:
        action_min = np.array(action_min_10)
        action_max = np.array(action_max_10)
    else:
        action_min = np.array(action_min_10)
        action_max = np.array(action_max_10)

    resize_size = get_image_resize_size(cfg)

    train_data_cfg = model.train_config['data']
    input_modality = train_data_cfg.get("input_modality", "image")
    view_mode = train_data_cfg.get("view_mode", "single")

    aug = getattr(getattr(cfg, "data", None), "augmentation", DictConfig({}))
    center_crop = bool(getattr(aug, "center_crop", getattr(cfg.eval, "center_crop", False)))
    center_crop_ratio = float(getattr(aug, "center_crop_ratio", 1.0))

    resume_episodes = int(getattr(cfg.eval, "resume_episodes", 0) or 0)
    resume_successes = int(getattr(cfg.eval, "resume_successes", 0) or 0)
    resume_successes = min(resume_successes, resume_episodes)
    episodes_seen = 0
    total_episodes, total_successes = resume_episodes, resume_successes
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        task = task_suite.get_task(task_id)

        task_category_raw = task_name_to_category.get(task.name, None)
        task_category = category_name_map.get(task_category_raw, None)

        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = get_libero_env(task, "vlanext", resolution=256)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.eval.num_trials_per_task)):
            if episodes_seen < resume_episodes:
                episodes_seen += 1
                continue
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            image_history = []
            image_history_wrist = []
            state_history = []  
            action_history = [] 
            action_buffer = []  

            if cfg.eval.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.eval.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.eval.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.eval.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.eval.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 400

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            while t < max_steps + cfg.eval.num_steps_wait:
                try:
                    if t < cfg.eval.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action("vlanext"))
                        t += 1
                        continue

                    img = get_libero_image(
                        obs,
                        resize_size,
                        center_crop=center_crop,
                        center_crop_ratio=center_crop_ratio,
                        obs_key="agentview_image",
                    )
                    image_history.append(img)

                    if view_mode == "multi":
                        img_wrist = get_libero_image(
                            obs,
                            resize_size,
                            center_crop=center_crop,
                            center_crop_ratio=center_crop_ratio,
                            obs_key="robot0_eye_in_hand_image",
                        )
                        image_history_wrist.append(img_wrist)
                    else:
                        img_wrist = None

                    replay_images.append(img)

                    gripper_state = np.clip(1 - (np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04), 0.0, 1.0)
                    
                    current_state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [gripper_state])
                    )
                    state_history.append(current_state)

                    observation = {
                        "full_image": img,
                        "full_image_wrist": img_wrist if img_wrist is not None else img,
                        "image_history": image_history if input_modality == "video" else [img],
                        "image_history_wrist": image_history_wrist if (input_modality == "video" and view_mode == "multi") else ([img_wrist] if view_mode == "multi" else []),
                        "state_history": state_history,
                        "action_history": action_history,
                    }

                    if len(action_buffer) == 0:
                        raw_action_chunk = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )
                        
                        if raw_action_chunk.ndim == 1:
                            raw_action_chunk = raw_action_chunk[None, :]
                            
                        steps_to_exec = getattr(cfg.eval, "num_steps_execute", 1)
                        steps_to_exec = min(steps_to_exec, len(raw_action_chunk))
                        
                        action_buffer = list(raw_action_chunk[:steps_to_exec])
                    
                    raw_action = action_buffer.pop(0)
                    action_history.append(raw_action)

                    action = raw_action.copy()
                    
                    action[:6] = (action[:6] + 1) / 2 * (action_max - action_min) + action_min
                    
                    action[6] = 1.0 if action[6] > 0 else -1.0

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        if task_category:
                            category_stats[task_category]['success'] += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            episodes_seen += 1
            if task_category:
                category_stats[task_category]['total'] += 1

            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, save_dir=str(eval_dir), fps=20
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")

            print("--- Category Success Rates ---")
            log_file.write("--- Category Success Rates ---\n")
            for cat, stats in category_stats.items():
                if stats['total'] > 0:
                    sr = stats['success'] / stats['total']
                    msg = f"{cat}: {stats['success']}/{stats['total']} ({sr*100:.1f}%)"
                    print(msg)
                    log_file.write(msg + "\n")

            log_file.flush()

        task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        print(f"Current task success rate: {task_sr}")
        print(f"Current total success rate: {total_sr}")
        log_file.write(f"Current task success rate: {task_sr}\n")
        log_file.write(f"Current total success rate: {total_sr}\n")
        log_file.flush()

        env.close()

    log_file.close()

    if total_episodes > 0:
        final_success_rate = (total_successes / total_episodes) * 100
        new_dir_name = f"{eval_dir.name}_SR{final_success_rate:.2f}"
        new_eval_dir = eval_dir.parent / new_dir_name
        
        try:
            print(f"Renaming output directory to include success rate: {new_eval_dir}")
            eval_dir.rename(new_eval_dir)
        except Exception as e:
            print(f"Failed to rename directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_plus_bench_config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = DictConfig(config_dict)
    eval_libero(cfg)
