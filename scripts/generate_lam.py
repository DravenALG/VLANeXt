import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.libero_act import normalize_image_resize_size, resize_frame_uint8
from src.datasets.libero_lerobot_act import MAIN_VIDEO_KEY, WRIST_VIDEO_KEY, _read_video_frames
from src.models.latent_action_models import load_lam_checkpoint


DEFAULT_SOURCE_ROOT = "/data/NTU_slab/draven/data/LIBERO_fastwam"
DEFAULT_OUTPUT_ROOT = "/data/NTU_slab/draven/data/LIBERO_fastwam_lam"
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def format_lerobot_path(root, template, episode_chunk, episode_index, video_key=None):
    values = {"episode_chunk": episode_chunk, "episode_index": episode_index}
    if video_key is not None:
        values["video_key"] = video_key
    return os.path.join(root, template.format(**values))


def to_tensor_image(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 127.5 - 1.0


def read_episode_frames(video_path, length, image_resize_size):
    frames = _read_video_frames(video_path, range(length))
    return [resize_frame_uint8(frames[i], image_resize_size) for i in range(length)]


def make_pair_tensor(main_a, main_b, wrist_a=None, wrist_b=None):
    frame_t = to_tensor_image(main_a)
    frame_tp1 = to_tensor_image(main_b)
    if wrist_a is not None and wrist_b is not None:
        frame_t = torch.cat([frame_t, to_tensor_image(wrist_a)], dim=0)
        frame_tp1 = torch.cat([frame_tp1, to_tensor_image(wrist_b)], dim=0)
    return frame_t, frame_tp1


def infer_episode_latents(model, device, main_frames, wrist_frames, batch_size):
    length = len(main_frames)
    latent_dim = int(model.latent_dim)
    if length <= 1:
        return np.zeros((length, latent_dim), dtype=np.float32)

    latents = []
    use_multi = wrist_frames is not None
    for start in range(0, length - 1, batch_size):
        end = min(length - 1, start + batch_size)
        frames_t = []
        frames_tp1 = []
        for idx in range(start, end):
            wrist_a = wrist_frames[idx] if use_multi else None
            wrist_b = wrist_frames[idx + 1] if use_multi else None
            frame_t, frame_tp1 = make_pair_tensor(
                main_frames[idx],
                main_frames[idx + 1],
                wrist_a=wrist_a,
                wrist_b=wrist_b,
            )
            frames_t.append(frame_t)
            frames_tp1.append(frame_tp1)
        frame_t = torch.stack(frames_t).to(device, non_blocking=True)
        frame_tp1 = torch.stack(frames_tp1).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            action = model.encode_action(frame_t, frame_tp1, deterministic=True)
        latents.append(action.float().cpu().numpy())

    latents = np.concatenate(latents, axis=0).astype(np.float32)
    last = latents[-1:]
    return np.concatenate([latents, last], axis=0).astype(np.float32)


def action_stats(actions):
    actions = np.asarray(actions, dtype=np.float32)
    return {
        "min": actions.min(axis=0).astype(float).tolist(),
        "max": actions.max(axis=0).astype(float).tolist(),
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0).astype(float).tolist(),
        "count": [int(actions.shape[0])],
    }


def update_info(info, *, latent_dim, checkpoint_path, lam_type, selected_episodes=None):
    action_feature = info.setdefault("features", {}).setdefault("action", {})
    action_feature["dtype"] = "float32"
    action_feature["shape"] = [int(latent_dim)]
    action_feature["names"] = {"latent_action": [f"latent_{i}" for i in range(int(latent_dim))]}
    action_feature["info"] = {
        "latent_action": True,
        "lam_type": lam_type,
        "lam_checkpoint": str(checkpoint_path),
    }
    if selected_episodes is not None:
        total_episodes = len(selected_episodes)
        total_frames = sum(int(ep["length"]) for ep in selected_episodes)
        video_features = [name for name, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["total_videos"] = total_episodes * len(video_features)
        info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def copy_dataset(source_root, output_root, overwrite):
    source_root = Path(source_root)
    output_root = Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source_root and output_root must be different")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def convert_suite(
    *,
    suite_root,
    checkpoint_path,
    model,
    device,
    view_mode,
    image_resize_size,
    batch_size,
    max_episodes,
):
    meta_dir = suite_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    with open(info_path, "r") as f:
        info = json.load(f)

    episodes = read_jsonl(episodes_path)
    selected_episodes = episodes[:max_episodes] if max_episodes is not None else episodes
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    stats_rows = {int(row["episode_index"]): row for row in read_jsonl(stats_path)} if stats_path.exists() else {}

    converted_stats = {}
    pbar = tqdm(selected_episodes, desc=f"convert {suite_root.name}")
    for episode in pbar:
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // chunks_size
        parquet_path = format_lerobot_path(suite_root, data_template, episode_chunk, episode_index)
        main_video_path = format_lerobot_path(suite_root, video_template, episode_chunk, episode_index, MAIN_VIDEO_KEY)
        wrist_video_path = format_lerobot_path(suite_root, video_template, episode_chunk, episode_index, WRIST_VIDEO_KEY)

        df = pd.read_parquet(parquet_path)
        if "frame_index" in df.columns:
            frame_indices = df["frame_index"].to_numpy()
            if len(frame_indices) > 1 and np.any(np.diff(frame_indices) < 0):
                df = df.sort_values("frame_index").reset_index(drop=True)
        length = len(df)
        main_frames = read_episode_frames(main_video_path, length, image_resize_size)
        wrist_frames = None
        if view_mode == "multi":
            wrist_frames = read_episode_frames(wrist_video_path, length, image_resize_size)

        latents = infer_episode_latents(model, device, main_frames, wrist_frames, batch_size)
        df["action"] = [latents[i].astype(np.float32) for i in range(length)]
        df.to_parquet(parquet_path, index=False)
        converted_stats[episode_index] = action_stats(latents)

    rows_to_write = selected_episodes if max_episodes is not None else episodes
    if max_episodes is not None:
        write_jsonl(episodes_path, selected_episodes)

    new_stats_rows = []
    for episode in rows_to_write:
        episode_index = int(episode["episode_index"])
        row = stats_rows.get(episode_index, {"episode_index": episode_index, "stats": {}})
        row.setdefault("stats", {})["action"] = converted_stats[episode_index]
        new_stats_rows.append(row)
    write_jsonl(stats_path, new_stats_rows)

    info = update_info(
        info,
        latent_dim=model.latent_dim,
        checkpoint_path=checkpoint_path,
        lam_type=model.lam_type,
        selected_episodes=selected_episodes if max_episodes is not None else None,
    )
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def discover_suites(root, requested):
    if requested:
        return tuple(requested)
    found = []
    for name in DEFAULT_SUITES:
        if (Path(root) / name / "meta" / "info.json").is_file():
            found.append(name)
    if found:
        return tuple(found)
    return tuple(sorted(path.name for path in Path(root).iterdir() if (path / "meta" / "info.json").is_file()))


def main(args):
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if args.copy_dataset:
        copy_dataset(source_root, output_root, overwrite=args.overwrite)
    elif not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist and --no-copy was set: {output_root}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, _ = load_lam_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    data_cfg = config.get("data", {})
    view_mode = args.view_mode or data_cfg.get("view_mode", "multi")
    image_resize_size = normalize_image_resize_size(args.image_resize_size or data_cfg.get("image_resize_size", 128))
    suites = discover_suites(output_root, args.suites)
    for suite in suites:
        convert_suite(
            suite_root=output_root / suite,
            checkpoint_path=args.checkpoint,
            model=model,
            device=device,
            view_mode=view_mode,
            image_resize_size=image_resize_size,
            batch_size=int(args.batch_size),
            max_episodes=args.max_episodes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a train_lam.py checkpoint.")
    parser.add_argument("--source-root", type=str, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-copy", dest="copy_dataset", action="store_false", help="Convert an already-copied output root in place.")
    parser.set_defaults(copy_dataset=True)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional per-suite episode cap for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--view-mode", type=str, choices=("single", "multi"), default=None)
    parser.add_argument("--image-resize-size", type=int, default=None)
    parser.add_argument("--suites", nargs="*", default=None)
    main(parser.parse_args())
