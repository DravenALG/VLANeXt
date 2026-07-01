import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image


WRIST_VIDEO_KEY = "observation.images.wrist_left"
EXTERIOR_VIDEO_KEYS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
)
VIDEO_KEYS = EXTERIOR_VIDEO_KEYS + (WRIST_VIDEO_KEY,)
FRAME_CACHE_DIR = "frame_cache"


def _format_lerobot_path(root, template, chunk_index, file_index, video_key=None):
    kwargs = {
        "chunk_index": int(chunk_index),
        "file_index": int(file_index),
        "episode_chunk": int(chunk_index),
        "episode_index": int(file_index),
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return root / template.format(**kwargs)


def _format_frame_cache_path(root, chunk_index, file_index, video_key, resize_size):
    h, w = resize_size
    return (
        root
        / FRAME_CACHE_DIR
        / f"resize_{h}x{w}"
        / video_key
        / f"chunk-{int(chunk_index):03d}"
        / f"file-{int(file_index):03d}.npy"
    )


def _to_uint8(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    return frame.astype(np.uint8)


def _resize_frame(frame, resize_size):
    frame = _to_uint8(frame)
    target_h, target_w = resize_size
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    image = Image.fromarray(frame)
    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _load_info(root):
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Droid LeRobot info.json not found: {info_path}")
    with info_path.open("r") as f:
        return json.load(f)


def _load_episode_metadata(root, video_keys):
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")

    columns = ["episode_index"]
    for key in video_keys:
        columns.extend([
            f"videos/{key}/chunk_index",
            f"videos/{key}/file_index",
            f"videos/{key}/to_timestamp",
        ])
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in episode_paths],
        ignore_index=True,
    )


def _iter_video_jobs(root, info, episodes_df, video_keys, resize_size):
    video_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    fps = float(info.get("fps", 15.0))

    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        to_col = f"videos/{video_key}/to_timestamp"
        grouped = episodes_df.groupby([chunk_col, file_col], sort=True)
        for (chunk_index, file_index), group in grouped:
            expected_frames = int(round(float(group[to_col].max()) * fps))
            if expected_frames <= 0:
                continue
            video_path = _format_lerobot_path(
                root,
                video_template,
                chunk_index=chunk_index,
                file_index=file_index,
                video_key=video_key,
            )
            cache_path = _format_frame_cache_path(
                root,
                chunk_index=chunk_index,
                file_index=file_index,
                video_key=video_key,
                resize_size=resize_size,
            )
            yield video_key, video_path, cache_path, expected_frames


def _write_video_cache(video_path, cache_path, expected_frames, resize_size):
    target_h, target_w = resize_size
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    frames_np = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.uint8,
        shape=(expected_frames, target_h, target_w, 3),
    )

    decoded = 0
    try:
        for frame in iio.imiter(video_path):
            if decoded >= expected_frames:
                break
            frames_np[decoded] = _resize_frame(frame, resize_size)
            decoded += 1
        frames_np.flush()
        del frames_np

        if decoded != expected_frames:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Frame count mismatch for {video_path}: decoded={decoded} expected={expected_frames}"
            )
        tmp_path.replace(cache_path)
    except Exception:
        try:
            del frames_np
        except UnboundLocalError:
            pass
        tmp_path.unlink(missing_ok=True)
        raise


def _parse_resize_size(value):
    if "x" in value.lower():
        h, w = value.lower().split("x", 1)
        return int(h), int(w)
    size = int(value)
    return size, size


def main():
    parser = argparse.ArgumentParser(description="Build resized frame caches for Droid LeRobot videos.")
    parser.add_argument(
        "root",
        nargs="?",
        default="/data/NTU_slab/draven/data/MolmoAct2-DROID",
        help="Path to the Droid LeRobot dataset root.",
    )
    parser.add_argument("--resize-size", default="256", help="Output frame size as INT or HxW. Default: 256.")
    parser.add_argument("--video-key", action="append", choices=VIDEO_KEYS, help="Video key to cache. Repeatable.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild caches that already exist.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    resize_size = _parse_resize_size(args.resize_size)
    video_keys = tuple(args.video_key) if args.video_key else VIDEO_KEYS

    info = _load_info(root)
    episodes_df = _load_episode_metadata(root, video_keys)

    built = 0
    skipped = 0
    missing = 0
    for job_index, (video_key, video_path, cache_path, expected_frames) in enumerate(
        _iter_video_jobs(root, info, episodes_df, video_keys, resize_size)
    ):
        if args.max_files is not None and job_index >= args.max_files:
            break
        if cache_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        if not video_path.is_file():
            missing += 1
            print(f"[skip] missing video: {video_path}")
            continue

        print(f"[build] {video_key} {video_path} frames={expected_frames}")
        _write_video_cache(video_path, cache_path, expected_frames, resize_size)
        built += 1
        print(f"[ok] {cache_path}")

    print(f"Done: built={built} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
