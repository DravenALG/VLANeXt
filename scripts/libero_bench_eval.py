"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

import argparse
import os
import time
from pathlib import Path

import tqdm
import wandb
import yaml
from libero.libero import benchmark

from src.datasets.libero_act import get_libero_action_stats, get_libero_normalization_suite_name
from src.evaluation.libero_bench.parallel_eval_utils import (
    build_episode_specs,
    chunked,
    rollout_episode_batch,
)
from src.evaluation.libero_bench.VLANeXt_utils import get_processor as get_vlanext_processor
from src.evaluation.libero_bench.libero_utils import save_rollout_video
from src.evaluation.libero_bench.robot_utils import (
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from src.utils.logging_utils import format_duration, progress_timing, setup_logger


def eval_libero(config) -> None:
    class DictConfig:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, DictConfig(v))
                else:
                    setattr(self, k, v)

    cfg = DictConfig(config)
    assert cfg.eval.finetuned_checkpoint is not None, "cfg.eval.finetuned_checkpoint must not be None!"

    set_seed_everywhere(cfg.eval.seed)
    model = get_model(cfg)
    processor = get_vlanext_processor(cfg)

    checkpoint_path = Path(cfg.eval.finetuned_checkpoint)
    try:
        step = str(checkpoint_path.stem.split("_")[-1])
    except ValueError:
        step = "unknown"

    num_steps_execute = int(getattr(cfg.eval, "num_steps_execute", 1))
    diffusion_steps = int(getattr(cfg.model, "diffusion_steps", -1))
    num_parallel_envs = max(1, int(getattr(cfg.eval, "num_parallel_envs", 1)))

    output_dir = checkpoint_path.parent
    eval_dir = output_dir / (
        f"{cfg.eval.task_suite_name}_checkpoint_{step}"
        f"_exec{num_steps_execute}"
        f"_diff{diffusion_steps}"
        f"_libero"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = getattr(getattr(cfg, "project", None), "use_wandb", True)
    if use_wandb:
        sub_dir = checkpoint_path.parent.name
        parent_dir = checkpoint_path.parent.parent.name
        wandb_project = f"{parent_dir}_testing"
        wandb_name = f"{sub_dir}_eval_{cfg.eval.task_suite_name}_step{step}_diff{diffusion_steps}_exec{num_steps_execute}"
        wandb.init(project=wandb_project, name=wandb_name, config=config)

    local_log_filepath = eval_dir / "eval.log"
    logger = setup_logger("libero_eval", local_log_filepath)
    logger.info("Logging to local log file: %s", local_log_filepath)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.eval.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info("Task suite: %s", cfg.eval.task_suite_name)

    train_config = model.train_config
    train_data_cfg = train_config["data"]
    train_suite_name = train_data_cfg["task_suite_name"]
    normalization_suite_name = get_libero_normalization_suite_name(
        train_suite_name,
        configured_suite_name=train_data_cfg.get("normalization_suite_name"),
    )
    action_min, action_max = get_libero_action_stats(normalization_suite_name)
    logger.info("Action normalization stats: %s", normalization_suite_name)

    resize_size = get_image_resize_size(cfg)
    input_modality = train_data_cfg.get("input_modality", "image")
    view_mode = train_data_cfg.get("view_mode", "single")

    aug = getattr(getattr(cfg, "data", None), "augmentation", DictConfig({}))
    center_crop = bool(getattr(aug, "center_crop", getattr(cfg.eval, "center_crop", False)))
    center_crop_ratio = float(getattr(aug, "center_crop_ratio", 1.0))

    resume_episodes = int(getattr(cfg.eval, "resume_episodes", 0) or 0)
    resume_successes = int(getattr(cfg.eval, "resume_successes", 0) or 0)
    resume_successes = min(resume_successes, resume_episodes)
    total_episodes, total_successes = resume_episodes, resume_successes
    planned_episodes = num_tasks_in_suite * int(cfg.eval.num_trials_per_task)
    remaining_episodes = max(0, planned_episodes - resume_episodes)
    eval_start_time = time.time()

    task_stats = {
        task_id: {"episodes": 0, "successes": 0}
        for task_id in range(num_tasks_in_suite)
    }
    specs = build_episode_specs(
        task_suite,
        num_tasks_in_suite,
        int(cfg.eval.num_trials_per_task),
        resume_episodes=resume_episodes,
    )

    logger.info(
        "Eval setup: checkpoint=%s tasks=%d trials_per_task=%d total_episodes=%d resume_episodes=%d remaining_episodes=%d parallel_envs=%d input_modality=%s view_mode=%s num_steps_execute=%d diffusion_steps=%d",
        checkpoint_path,
        num_tasks_in_suite,
        cfg.eval.num_trials_per_task,
        planned_episodes,
        resume_episodes,
        remaining_episodes,
        num_parallel_envs,
        input_modality,
        view_mode,
        num_steps_execute,
        diffusion_steps,
    )

    progress = tqdm.tqdm(total=len(specs), disable=not os.isatty(2))
    for batch_specs in chunked(specs, num_parallel_envs):
        results = rollout_episode_batch(
            cfg,
            model,
            processor,
            batch_specs,
            action_min,
            action_max,
            resize_size,
            center_crop,
            center_crop_ratio,
            input_modality,
            view_mode,
            logger,
        )

        for result in results:
            spec = result.spec
            task_stat = task_stats[spec.task_id]
            task_stat["episodes"] += 1
            total_episodes += 1
            if result.success:
                task_stat["successes"] += 1
                total_successes += 1

            completed_since_resume = max(0, total_episodes - resume_episodes)
            elapsed, avg_episode_time, eta = progress_timing(
                eval_start_time,
                completed_since_resume,
                remaining_episodes,
            )
            video_path = save_rollout_video(
                result.replay_images,
                spec.global_episode_id,
                success=result.success,
                task_description=spec.task_description,
                logger=logger,
                save_dir=str(eval_dir),
                fps=20,
            )

            task_sr = (
                float(task_stat["successes"]) / float(task_stat["episodes"])
                if task_stat["episodes"] > 0 else 0.0
            )
            total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
            logger.info(
                "Episode result: task=%d/%d trial=%d success=%s rollout_steps=%d episode_time=%s total_episodes=%d/%d total_successes=%d task_success_rate=%.1f%% total_success_rate=%.1f%% elapsed=%s avg_episode=%.2fs eta=%s video=%s",
                spec.task_id + 1,
                num_tasks_in_suite,
                spec.episode_idx,
                result.success,
                result.rollout_steps,
                format_duration(result.episode_time),
                total_episodes,
                planned_episodes,
                total_successes,
                task_sr * 100,
                total_sr * 100,
                format_duration(elapsed),
                avg_episode_time,
                format_duration(eta),
                video_path,
            )

            if use_wandb:
                wandb_log = {
                    "eval/episode": total_episodes,
                    "eval/success": int(result.success),
                    "eval/total_success_rate": total_successes / total_episodes * 100,
                }
                if video_path and os.path.exists(video_path):
                    wandb_log["eval/rollout_video"] = wandb.Video(video_path, fps=20, format="mp4")
                wandb.log(wandb_log, step=total_episodes)

            progress.update(1)
    progress.close()

    for task_id in range(num_tasks_in_suite):
        stats = task_stats[task_id]
        if stats["episodes"] == 0:
            continue
        task_sr = float(stats["successes"]) / float(stats["episodes"])
        total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        logger.info(
            "Task summary: task=%d/%d task_successes=%d/%d task_success_rate=%.4f total_successes=%d/%d total_success_rate=%.4f",
            task_id + 1,
            num_tasks_in_suite,
            stats["successes"],
            stats["episodes"],
            task_sr,
            total_successes,
            total_episodes,
            total_sr,
        )

    if total_episodes > 0:
        final_success_rate = (total_successes / total_episodes) * 100
        new_dir_name = f"{eval_dir.name}_SR{final_success_rate:.2f}"
        new_eval_dir = eval_dir.parent / new_dir_name
        logger.info(
            "Eval finished: total_episodes=%d total_successes=%d final_success_rate=%.2f%% elapsed=%s",
            total_episodes,
            total_successes,
            final_success_rate,
            format_duration(time.time() - eval_start_time),
        )

        try:
            logger.info("Renaming output directory to include success rate: %s", new_eval_dir)
            eval_dir.rename(new_eval_dir)
        except Exception as e:
            logger.warning("Failed to rename directory: %s", e)

        if use_wandb:
            wandb.log({
                "eval/final_success_rate": final_success_rate,
                "eval/total_episodes": total_episodes,
                "eval/total_successes": total_successes,
            })
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/libero_bench_config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    eval_libero(config)
