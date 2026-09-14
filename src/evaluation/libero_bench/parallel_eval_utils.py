import time
from dataclasses import dataclass

import numpy as np

from src.evaluation.libero_bench.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_vector_env,
    quat2axisangle,
)
from src.evaluation.libero_bench.robot_utils import get_action


@dataclass
class EpisodeSpec:
    task_id: int
    episode_idx: int
    global_episode_id: int
    task: object
    init_state: object
    task_description: str
    task_category: str = None


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    success: bool
    rollout_steps: int
    episode_time: float
    replay_images: list


def get_libero_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    return 400


def build_episode_specs(task_suite, num_tasks, num_trials_per_task, resume_episodes=0, category_fn=None):
    specs = []
    seen = 0
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language
        task_category = category_fn(task) if category_fn is not None else None

        for episode_idx in range(num_trials_per_task):
            global_episode_id = seen + 1
            seen += 1
            if global_episode_id <= resume_episodes:
                continue
            specs.append(
                EpisodeSpec(
                    task_id=task_id,
                    episode_idx=episode_idx,
                    global_episode_id=global_episode_id,
                    task=task,
                    init_state=initial_states[episode_idx],
                    task_description=task_description,
                    task_category=task_category,
                )
            )
    return specs


def chunked(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _obs_at(batch_obs, idx):
    if isinstance(batch_obs, dict):
        return {key: value[idx] for key, value in batch_obs.items()}
    return batch_obs[idx]


def _as_done(value) -> bool:
    return bool(np.asarray(value).item())


def _get_normalized_zero_action(action_min, action_max, action_dim: int, gripper: float = -1.0):
    action_min = np.asarray(action_min, dtype=np.float32)
    action_max = np.asarray(action_max, dtype=np.float32)
    denominator = action_max - action_min
    denominator = np.where(denominator == 0, 1.0, denominator)

    normalized_zero_pose = 2.0 * (np.zeros_like(action_min) - action_min) / denominator - 1.0
    normalized_zero_pose = np.clip(normalized_zero_pose, -1.0, 1.0)
    action = np.concatenate([normalized_zero_pose, np.array([gripper], dtype=np.float32)])
    if len(action) != int(action_dim):
        raise ValueError(f"Expected LIBERO action_dim={len(action)}, got {action_dim}")
    return action


def _make_eval_observation(
    obs,
    state,
    resize_size,
    center_crop,
    center_crop_ratio,
    input_modality,
    view_mode,
):
    img = get_libero_image(
        obs,
        resize_size,
        center_crop=center_crop,
        center_crop_ratio=center_crop_ratio,
        obs_key="agentview_image",
    )
    state["image_history"].append(img)
    state["replay_images"].append(img)

    if view_mode == "multi":
        img_wrist = get_libero_image(
            obs,
            resize_size,
            center_crop=center_crop,
            center_crop_ratio=center_crop_ratio,
            obs_key="robot0_eye_in_hand_image",
        )
        state["image_history_wrist"].append(img_wrist)
    else:
        img_wrist = None

    gripper_state = np.clip(1 - (np.mean(np.abs(obs["robot0_gripper_qpos"])) / 0.04), 0.0, 1.0)
    current_state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), [gripper_state])
    )
    state["state_history"].append(current_state)

    return {
        "full_image": img,
        "full_image_wrist": img_wrist if img_wrist is not None else img,
        "image_history": state["image_history"] if input_modality == "video" else [img],
        "image_history_wrist": (
            state["image_history_wrist"]
            if (input_modality == "video" and view_mode == "multi")
            else ([img_wrist] if view_mode == "multi" else [])
        ),
        "state_history": state["state_history"],
        "action_history": state["action_history"],
        "history_action_pad": state["history_action_pad"],
    }


def _denormalize_action(raw_action, action_min, action_max):
    action = raw_action.copy()
    action[:6] = (action[:6] + 1) / 2 * (action_max - action_min) + action_min
    action[6] = 1.0 if action[6] > 0 else -1.0
    return action


def rollout_episode_batch(
    cfg,
    model,
    processor,
    specs,
    action_min,
    action_max,
    resize_size,
    center_crop,
    center_crop_ratio,
    input_modality,
    view_mode,
    logger,
):
    env, _ = get_libero_vector_env([spec.task for spec in specs], resolution=cfg.eval.image_size)
    max_steps = get_libero_max_steps(cfg.eval.task_suite_name)
    num_steps_wait = int(cfg.eval.num_steps_wait)
    num_steps_execute = int(getattr(cfg.eval, "num_steps_execute", 1))
    num_envs = len(specs)
    history_action_pad = _get_normalized_zero_action(
        action_min,
        action_max,
        getattr(model, "action_dim", 7),
    )

    states = [
        {
            "t": 0,
            "done": False,
            "success": False,
            "start_time": time.time(),
            "replay_images": [],
            "image_history": [],
            "image_history_wrist": [],
            "state_history": [],
            "action_history": [],
            "history_action_pad": history_action_pad,
            "action_buffer": [],
        }
        for _ in specs
    ]
    obs_by_env = [None] * num_envs

    try:
        env.reset()
        init_states = [spec.init_state for spec in specs]
        batch_obs = env.set_init_state(init_states)
        obs_by_env = [_obs_at(batch_obs, idx) for idx in range(num_envs)]

        logger.info(
            "Starting parallel batch: episodes=%s max_steps=%d",
            [spec.global_episode_id for spec in specs],
            max_steps,
        )

        while True:
            live_ids = [
                idx for idx, state in enumerate(states)
                if not state["done"] and state["t"] < max_steps + num_steps_wait
            ]
            if not live_ids:
                break

            wait_ids = [idx for idx in live_ids if states[idx]["t"] < num_steps_wait]
            if wait_ids:
                dummy = np.asarray([get_libero_dummy_action("vlanext") for _ in wait_ids])
                batch_obs, _, batch_done, _ = env.step(dummy, id=wait_ids)
                for local_idx, env_id in enumerate(wait_ids):
                    obs_by_env[env_id] = _obs_at(batch_obs, local_idx)
                    states[env_id]["t"] += 1
                    if _as_done(batch_done[local_idx]):
                        states[env_id]["done"] = True
                        states[env_id]["success"] = True
                continue

            action_request_ids = []
            action_request_obs = []
            action_request_labels = []

            for env_id in live_ids:
                observation = _make_eval_observation(
                    obs_by_env[env_id],
                    states[env_id],
                    resize_size,
                    center_crop,
                    center_crop_ratio,
                    input_modality,
                    view_mode,
                )
                if not states[env_id]["action_buffer"]:
                    action_request_ids.append(env_id)
                    action_request_obs.append(observation)
                    action_request_labels.append(specs[env_id].task_description)

            if action_request_ids:
                raw_chunks = get_action(
                    cfg,
                    model,
                    action_request_obs,
                    action_request_labels,
                    processor=processor,
                )
                if raw_chunks.ndim == 2:
                    raw_chunks = raw_chunks[:, None, :]

                for batch_idx, env_id in enumerate(action_request_ids):
                    chunk = raw_chunks[batch_idx]
                    steps_to_exec = min(num_steps_execute, len(chunk))
                    states[env_id]["action_buffer"] = list(chunk[:steps_to_exec])

            step_ids = [
                idx for idx in live_ids
                if states[idx]["action_buffer"] and not states[idx]["done"]
            ]
            if not step_ids:
                break

            actions = []
            for env_id in step_ids:
                raw_action = states[env_id]["action_buffer"].pop(0)
                states[env_id]["action_history"].append(raw_action)
                actions.append(_denormalize_action(raw_action, action_min, action_max))

            batch_obs, _, batch_done, _ = env.step(np.asarray(actions), id=step_ids)
            for local_idx, env_id in enumerate(step_ids):
                obs_by_env[env_id] = _obs_at(batch_obs, local_idx)
                states[env_id]["t"] += 1
                if _as_done(batch_done[local_idx]):
                    states[env_id]["done"] = True
                    states[env_id]["success"] = True

    except Exception:
        logger.exception("Caught exception during parallel rollout batch")
    finally:
        env.close()

    return [
        EpisodeResult(
            spec=spec,
            success=state["success"],
            rollout_steps=len(state["action_history"]),
            episode_time=time.time() - state["start_time"],
            replay_images=state["replay_images"],
        )
        for spec, state in zip(specs, states)
    ]
