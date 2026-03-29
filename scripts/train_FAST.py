"""
Train a FAST (Frequency-space Action Sequence Tokenizer) on a Libero dataset.

Usage:
    python scripts/train_FAST.py \
        --config config/libero_train_config.yaml \
        --output_dir /path/to/save/fast_tokenizer \
        [--max_trajs 500] [--scale 10.0] [--vocab_size 2048]

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

from src.models.FAST_ActionTokenizer.processing_action_tokenizer import UniversalActionProcessor
from src.datasets.libero_act import (
    action_min_spatial, action_max_spatial,
    action_min_object, action_max_object,
    action_min_goal, action_max_goal,
    action_min_10, action_max_10,
)


def get_action_stats(task_suite_name: str):
    if 'spatial' in task_suite_name:
        return np.array(action_min_spatial), np.array(action_max_spatial)
    elif 'object' in task_suite_name:
        return np.array(action_min_object), np.array(action_max_object)
    elif 'goal' in task_suite_name:
        return np.array(action_min_goal), np.array(action_max_goal)
    else:
        return np.array(action_min_10), np.array(action_max_10)


def collect_action_chunks(
    data_path: str,
    future_len: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    max_trajs: int | None = None,
) -> list[np.ndarray]:
    """
    Stream the dataset and collect normalized action chunks of shape (future_len, action_dim).
    Normalization follows the same convention as LiberoAct:
      - delta_pose (dims 0-5): scaled to [-1, 1] using dataset min/max
      - gripper  (dim 6)     : clipped to [-1, 1]
    End-of-trajectory chunks are edge-padded with the last action.
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

    action_chunks: list[np.ndarray] = []

    for traj_id, traj_data in tqdm(enumerate(ds), total=total, desc="Collecting chunks"):
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

            # Build one chunk per timestep
            for t in range(traj_len):
                end = min(t + future_len, traj_len)
                chunk = actions_np[t:end]
                if chunk.shape[0] < future_len:
                    # Edge-pad with the last action
                    pad = np.tile(actions_np[-1:], (future_len - chunk.shape[0], 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                action_chunks.append(chunk)  # (future_len, action_dim)

        except Exception as e:
            print(f"[Warn] Skipping trajectory {traj_id}: {e}")
            continue

    return action_chunks


def main():
    parser = argparse.ArgumentParser(description="Train FAST action tokenizer on Libero data.")
    parser.add_argument('--config', type=str, default='config/libero_train_config.yaml',
                        help='Path to the training YAML config')
    parser.add_argument('--output_dir', type=str, default='src/models/FAST_ActionTokenizer/libero_spatial',
                        help='Directory to save the trained tokenizer')
    parser.add_argument('--max_trajs', type=int, default=None,
                        help='Cap on the number of trajectories to use (default: all)')
    parser.add_argument('--scale', type=float, default=10.0,
                        help='DCT quantization scale (default: 10.0)')
    parser.add_argument('--vocab_size', type=int, default=2048,
                        help='BPE vocabulary size (default: 2048)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = config['data']['data_root']
    task_suite = config['data']['task_suite_name']
    future_len = config['data']['future_len']
    action_dim = config['model']['action_dim']
    data_path = os.path.join(data_root, task_suite, "1.0.0")

    print(f"Dataset  : {data_path}")
    print(f"future_len={future_len}  action_dim={action_dim}")
    print(f"scale={args.scale}  vocab_size={args.vocab_size}")

    action_min, action_max = get_action_stats(task_suite)

    # ------------------------------------------------------------------ #
    # 1. Collect chunks
    # ------------------------------------------------------------------ #
    action_chunks = collect_action_chunks(
        data_path=data_path,
        future_len=future_len,
        action_min=action_min,
        action_max=action_max,
        max_trajs=args.max_trajs,
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
        scale=args.scale,
        vocab_size=args.vocab_size,
        time_horizon=future_len,
        action_dim=action_dim,
    )

    # ------------------------------------------------------------------ #
    # 3. Save
    #    ProcessorMixin.save_pretrained would put the BPE tokenizer into a
    #    `bpe_tokenizer/` subdirectory, but VLANeXt loads it directly from
    #    the top-level path.  So we save the BPE tokenizer flat instead.
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    processor.bpe_tokenizer.save_pretrained(args.output_dir)

    # Write the canonical processor_config.json expected by VLANeXt
    proc_cfg = {
        "processor_class": "UniversalActionProcessor",
        "auto_map": {
            "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
        },
        "scale": args.scale,
        "vocab_size": args.vocab_size,
        "min_token": processor.min_token,
        "action_dim": action_dim,
        "time_horizon": future_len,
    }
    cfg_path = os.path.join(args.output_dir, "processor_config.json")
    with open(cfg_path, "w") as f:
        json.dump(proc_cfg, f, indent=2)

    print(f"\nTokenizer saved to: {args.output_dir}")
    print(f"  min_token = {processor.min_token}")
    print(f"  vocab_size = {args.vocab_size}")
    print(f"  scale = {args.scale}")
    print(f"\nTo use this tokenizer, set in your config:")
    print(f"  model.fast_action_tokenizer.enabled: true")
    print(f"  model.fast_action_tokenizer.tokenizer_path: \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
