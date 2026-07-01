"""
Train a FAST (Frequency-space Action Sequence Tokenizer) on a Libero dataset.

Usage:
    python scripts/train_FAST.py --config config/libero_train_fast_config.yaml

Set the suite, output directory, and tokenizer hyperparameters in the config:
    data.task_suite_name
    data.normalization_suite_name
    fast.output_dir
    fast.max_trajs
    fast.scale
    fast.vocab_size

The saved directory can then be referenced in the training config via:
    model.fast_action_tokenizer.tokenizer_path: "/path/to/save/fast_tokenizer"
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

from src.models.fast_tokenizer.processing_action_tokenizer import UniversalActionProcessor
from src.datasets.libero_act import LiberoMixedAct, get_libero_action_stats


def resolve_suite_paths(
    data_root: str,
    task_suite_name: str,
    version: str = "1.0.0",
) -> list[tuple[str, str]]:
    suites = LiberoMixedAct.SUITES if task_suite_name == "libero_mixed" else (task_suite_name,)
    return [(suite, os.path.join(data_root, suite, version)) for suite in suites]


def collect_action_chunks(
    data_path: str,
    future_len: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    max_trajs: int | None = None,
    desc: str = "Collecting chunks",
) -> list[np.ndarray]:
    """
    Stream the dataset and collect normalized action chunks of shape (future_len, action_dim).
    Normalization follows the same convention as LiberoAct:
      - delta_pose (dims 0-5): scaled to [-1, 1] using dataset min/max
      - gripper  (dim 6)     : clipped to [-1, 1]
    End-of-trajectory chunks use the same future-action padding as LiberoAct.
    """
    builder = tfds.builder_from_directory(builder_dir=data_path)
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)

    total_in_split = builder.info.splits['train'].num_examples
    total = min(total_in_split, max_trajs) if max_trajs is not None else total_in_split
    if max_trajs is not None:
        ds = ds.take(max_trajs)

    action_denominator = action_max - action_min
    action_denominator = np.where(action_denominator == 0, 1.0, action_denominator)

    zero_pose = np.zeros_like(action_min, dtype=np.float32)
    normalized_zero_pose = 2.0 * (zero_pose - action_min) / action_denominator - 1.0
    normalized_zero_pose = np.clip(normalized_zero_pose, -1.0, 1.0)

    action_chunks: list[np.ndarray] = []

    for traj_id, traj_data in tqdm(enumerate(ds), total=total, desc=desc):
        try:
            traj_batch = next(iter(traj_data['steps'].batch(2000)))

            if traj_batch['reward'][-1].numpy() != 1:
                continue

            raw_actions = traj_batch['action'].numpy().astype(np.float32)
            traj_len = raw_actions.shape[0]

            # Normalize — same as LiberoAct
            delta_pose = raw_actions[:, :6]
            delta_pose = 2.0 * (delta_pose - action_min) / action_denominator - 1.0
            delta_pose = np.clip(delta_pose, -1.0, 1.0)

            gripper = raw_actions[:, 6:7]
            gripper = np.clip(gripper, -1.0, 1.0)

            actions_np = np.concatenate([delta_pose, gripper], axis=1)  # (T, action_dim)
            pad_action = np.concatenate([normalized_zero_pose, actions_np[-1, 6:7]])

            # Build one chunk per timestep
            for t in range(traj_len):
                end = min(t + future_len, traj_len)
                chunk = actions_np[t:end]
                if chunk.shape[0] < future_len:
                    pad = np.tile(pad_action[None], (future_len - chunk.shape[0], 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                action_chunks.append(chunk)  # (future_len, action_dim)

        except Exception as e:
            print(f"[Warn] Skipping trajectory {traj_id}: {e}")
            continue

    return action_chunks


def main():
    parser = argparse.ArgumentParser(description="Train FAST action tokenizer on Libero data.")
    parser.add_argument('--config', type=str, default='config/libero_train_fast_config.yaml',
                        help='Path to the FAST tokenizer training YAML config')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = config['data']['data_root']
    version = config['data'].get('version', '1.0.0')
    task_suite = config['data']['task_suite_name']
    stats_suite = config['data'].get('normalization_suite_name') or task_suite
    future_len = config['data']['future_len']
    action_dim = config['model']['action_dim']
    fast_config = config['fast']
    output_dir = fast_config['output_dir']
    max_trajs = fast_config.get('max_trajs')
    scale = fast_config['scale']
    vocab_size = fast_config['vocab_size']
    suite_paths = resolve_suite_paths(data_root, task_suite, version=version)

    print(f"Dataset  : {task_suite}")
    for suite_name, data_path in suite_paths:
        print(f"  - {suite_name}: {data_path}")
    print(f"stats    : {stats_suite}")
    print(f"future_len={future_len}  action_dim={action_dim}")
    print(f"scale={scale}  vocab_size={vocab_size}")

    action_min, action_max = get_libero_action_stats(stats_suite)

    # ------------------------------------------------------------------ #
    # 1. Collect chunks
    # ------------------------------------------------------------------ #
    action_chunks = []
    for suite_name, data_path in suite_paths:
        action_chunks.extend(
            collect_action_chunks(
                data_path=data_path,
                future_len=future_len,
                action_min=action_min,
                action_max=action_max,
                max_trajs=max_trajs,
                desc=f"Collecting {suite_name}",
            )
        )

    if not action_chunks:
        raise RuntimeError("No action chunks collected. Check your data path and dataset.")

    print(f"\nCollected {len(action_chunks)} chunks — shape {action_chunks[0].shape}")

    # ------------------------------------------------------------------ #
    # 2. Fit the tokenizer
    # ------------------------------------------------------------------ #
    print("\nTraining FAST tokenizer (BPE on DCT coefficients) ...")
    processor = UniversalActionProcessor.fit(
        action_data=action_chunks,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=future_len,
        action_dim=action_dim,
    )

    # ------------------------------------------------------------------ #
    # 3. Save
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    processor.bpe_tokenizer.save_pretrained(output_dir)

    # Write the canonical processor_config.json expected by VLANeXt
    proc_cfg = {
        "processor_class": "UniversalActionProcessor",
        "auto_map": {
            "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
        },
        "scale": scale,
        "vocab_size": vocab_size,
        "min_token": processor.min_token,
        "action_dim": action_dim,
        "time_horizon": future_len,
    }
    cfg_path = os.path.join(output_dir, "processor_config.json")
    with open(cfg_path, "w") as f:
        json.dump(proc_cfg, f, indent=2)

    print(f"\nTokenizer saved to: {output_dir}")
    print(f"  min_token = {processor.min_token}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  scale = {scale}")
    print(f"\nTo use this tokenizer, set in your config:")
    print(f"  model.fast_action_tokenizer.enabled: true")
    print(f"  model.fast_action_tokenizer.tokenizer_path: \"{output_dir}\"")


if __name__ == "__main__":
    main()
