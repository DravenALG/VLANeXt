import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


MAIN_VIDEO_KEY = "observation.images.image"
WRIST_VIDEO_KEY = "observation.images.wrist_image"
VIDEO_KEYS = (MAIN_VIDEO_KEY, WRIST_VIDEO_KEY)
FRAME_CACHE_DIR = "frame_cache"


def _read_jsonl(path):
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_lerobot_path(root, template, episode_chunk, episode_index, video_key=None):
    kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return root / template.format(**kwargs)


def _format_frame_cache_path(root, episode_chunk, episode_index, video_key, resize_size):
    h, w = resize_size
    return (
        root
        / FRAME_CACHE_DIR
        / f"resize_{h}x{w}"
        / f"chunk-{episode_chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.npy"
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


def _read_video(video_path, resize_size):
    frames = [_resize_frame(frame, resize_size) for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def _save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, array)
    tmp_path.replace(path)


def _find_suite_roots(root):
    if (root / "meta" / "info.json").is_file():
        return [root]
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    ]


def _parse_resize_size(value):
    if "x" in value.lower():
        h, w = value.lower().split("x", 1)
        return int(h), int(w)
    size = int(value)
    return size, size


def build_suite_cache(suite_root, resize_size):
    info_path = suite_root / "meta" / "info.json"
    episodes_path = suite_root / "meta" / "episodes.jsonl"

    with info_path.open("r") as f:
        info = json.load(f)
    episodes = _read_jsonl(episodes_path)

    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunks_size = int(info.get("chunks_size", 1000))

    print(f"Building frame cache for {suite_root}")
    built = 0
    skipped = 0
    missing = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // chunks_size
        expected_len = int(episode.get("length", 0) or 0)

        for video_key in VIDEO_KEYS:
            video_path = _format_lerobot_path(
                suite_root,
                video_template,
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            cache_path = _format_frame_cache_path(
                suite_root,
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
                resize_size=resize_size,
            )

            if cache_path.is_file():
                skipped += 1
                continue
            if not video_path.is_file():
                missing += 1
                print(f"[skip] missing video: {video_path}")
                continue

            frames = _read_video(video_path, resize_size)
            if expected_len and frames.shape[0] != expected_len:
                print(
                    f"[warn] frame count mismatch for {video_path}: "
                    f"decoded={frames.shape[0]} expected={expected_len}"
                )
            _save_npy(cache_path, frames)
            built += 1
            print(f"[ok] {cache_path} shape={frames.shape}")

    print(
        f"Done {suite_root.name}: built={built} skipped={skipped} "
        f"missing={missing}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build resized frame caches for LeRobot LIBERO videos.")
    parser.add_argument(
        "root",
        help="Path to the LeRobot LIBERO dataset root.",
    )
    parser.add_argument(
        "--resize-size",
        default="256",
        help="Output frame size as INT or HxW. Default: 256.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    resize_size = _parse_resize_size(args.resize_size)
    suite_roots = _find_suite_roots(root)
    if not suite_roots:
        raise RuntimeError(f"No LeRobot suite roots found under {root}")

    for suite_root in suite_roots:
        build_suite_cache(suite_root, resize_size)


if __name__ == "__main__":
    main()
