import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from src.datasets.language_action import (
    compute_padded_movement_actions,
    format_language_action_text,
    normalize_language_action_format,
)


action_min_droid = [-0.2077193409204483, -0.8572492003440857, -0.28132379055023193, -3.141592502593994, -1.570475697517395, -3.1415903568267822]
action_max_droid = [0.9295652508735657, 0.8648782968521118, 1.074978232383728, 3.1415927410125732, 1.569351077079773, 3.1415891647338867]

ACTION_KEY = "action.original"
STATE_POSE_KEY = "observation.state.cartesian_position"
STATE_GRIPPER_KEY = "observation.state.gripper_position"
WRIST_VIDEO_KEY = "observation.images.wrist_left"
EXTERIOR_VIDEO_KEYS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
)
VIDEO_KEYS = EXTERIOR_VIDEO_KEYS + (WRIST_VIDEO_KEY,)
FRAME_CACHE_DIR = "frame_cache"

DATA_COLUMNS = [
    "episode_index",
    "frame_index",
    "task_index",
    "language_instruction",
    STATE_POSE_KEY,
    STATE_GRIPPER_KEY,
    ACTION_KEY,
]


def normalize_image_resize_size(image_resize_size):
    if image_resize_size is None:
        return None
    if isinstance(image_resize_size, bool):
        raise ValueError("image_resize_size must be null, an int, or [height, width].")
    if isinstance(image_resize_size, int):
        if image_resize_size <= 0:
            raise ValueError(f"image_resize_size must be positive, got {image_resize_size}")
        return (image_resize_size, image_resize_size)
    if isinstance(image_resize_size, (list, tuple)) and len(image_resize_size) == 2:
        h, w = int(image_resize_size[0]), int(image_resize_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_resize_size must be positive, got {image_resize_size}")
        return (h, w)
    raise ValueError(f"image_resize_size must be null, an int, or [height, width], got {image_resize_size}")


def resize_frame_uint8(frame, resize_size):
    if resize_size is None:
        return frame
    target_h, target_w = resize_size
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    image = Image.fromarray(frame)
    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _format_lerobot_path(root, template, chunk_index, file_index, video_key=None):
    kwargs = {
        "chunk_index": int(chunk_index),
        "file_index": int(file_index),
        "episode_chunk": int(chunk_index),
        "episode_index": int(file_index),
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return os.path.join(root, template.format(**kwargs))


def _format_frame_cache_path(root, chunk_index, file_index, video_key, image_resize_size):
    if image_resize_size is None:
        return None
    h, w = image_resize_size
    return os.path.join(
        root,
        FRAME_CACHE_DIR,
        f"resize_{h}x{w}",
        video_key,
        f"chunk-{int(chunk_index):03d}",
        f"file-{int(file_index):03d}.npy",
    )


def _to_uint8(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    return frame.astype(np.uint8)


def _read_video_frames(video_path, frame_indices):
    """Read selected file-local RGB frames from a Droid LeRobot mp4."""
    unique_indices = sorted({int(i) for i in frame_indices})
    frames = {}
    for idx in unique_indices:
        frames[idx] = _to_uint8(iio.imread(video_path, index=idx))
    return frames


def _read_frame_cache_frames(cache_path, frame_indices):
    unique_indices = sorted({int(i) for i in frame_indices})
    if not unique_indices:
        return {}
    if not cache_path or not os.path.isfile(cache_path):
        return None

    frames_np = np.load(cache_path, mmap_mode="r")
    max_index = unique_indices[-1]
    if max_index >= frames_np.shape[0]:
        raise IndexError(
            f"Frame cache {cache_path} has {frames_np.shape[0]} frames, "
            f"but frame {max_index} was requested."
        )
    return {
        idx: np.asarray(frames_np[idx], dtype=np.uint8).copy()
        for idx in unique_indices
    }


def _stack_column(series, width=None):
    values = series.to_numpy()
    array = np.stack(values).astype(np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if width is not None and array.shape[1] != width:
        raise ValueError(f"Expected width {width}, got array shape {array.shape}.")
    return array


def _first_task(tasks):
    if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) > 0:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return ""


def _is_valid_text(text):
    if text is None:
        return False
    text = str(text).strip()
    return bool(text) and text.lower() != "nan"


def _binarize_open_gripper_from_close_position(gripper_position, threshold=0.5):
    """Match LAP Droid gripper labels: raw 0=open/1=close -> 1=open/0=close."""
    close_position = np.asarray(gripper_position, dtype=np.float32)
    if close_position.ndim == 1:
        close_position = close_position[:, None]
    open_position = 1.0 - np.clip(close_position, 0.0, 1.0)
    out = np.zeros_like(open_position, dtype=np.float32)
    open_mask = open_position > float(threshold)
    closed_mask = open_position < (1.0 - float(threshold))
    in_between = ~(open_mask | closed_mask)
    carry = open_position[-1] > float(threshold)
    for i in range(open_position.shape[0] - 1, -1, -1):
        update = ~in_between[i]
        carry = np.where(update, open_mask[i], carry)
        out[i] = carry.astype(np.float32)
    return out


class DroidLeRobotAct(IterableDataset):
    def __init__(
        self,
        data_path,
        dataset_name="droid",
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=True,
        input_modality="video",
        view_mode="single",
        load_future_image=False,
        future_image_mode="horizon",
        load_future_video=False,
        future_video_downsample=1,
        buffer_size=10000,
        sampling_rate=0.1,
        allow_end_padding=True,
        emit_proprioception=True,
        emit_history_actions=True,
        batch_size=1,
        drop_last=True,
        episode_cache_size=64,
        data_file_cache_size=2,
        seed=0,
        image_resize_size=None,
        load_language_action=False,
        language_action_format="vla-0",
        language_action_num_bins=1000,
        action_mode="droid",
        fps=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.full_sequence = bool(full_sequence)
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = bool(load_future_image)
        self.future_image_mode = future_image_mode
        self.load_future_video = bool(load_future_video)
        self.future_video_downsample = int(future_video_downsample)
        self.buffer_size = buffer_size
        self.sampling_rate = float(sampling_rate)
        self.allow_end_padding = bool(allow_end_padding)
        self.emit_proprioception = bool(emit_proprioception)
        self.emit_history_actions = bool(emit_history_actions)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.episode_cache_size = int(episode_cache_size)
        self.data_file_cache_size = int(data_file_cache_size)
        self.seed = int(seed)
        self.image_resize_size = normalize_image_resize_size(image_resize_size)
        self.load_language_action = bool(load_language_action)
        self.language_action_format = normalize_language_action_format(language_action_format)
        self.language_action_num_bins = int(language_action_num_bins)
        self.action_mode = str(action_mode or "droid").lower()
        self.fps = None if fps is None else float(fps)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.input_modality not in {"image", "video"}:
            raise ValueError(f"Unknown input_modality: {self.input_modality}")
        if self.view_mode not in {"single", "multi"}:
            raise ValueError(f"Unknown view_mode: {self.view_mode}")
        if self.future_video_downsample <= 0:
            raise ValueError(f"future_video_downsample must be positive, got {future_video_downsample}.")
        if self.load_future_video and self.future_len % self.future_video_downsample != 0:
            raise ValueError("future_len must be divisible by future_video_downsample.")
        if self.action_mode not in {"droid", "cartesian"}:
            raise ValueError("DroidLeRobotAct only supports action_mode='droid'/'cartesian'.")

        self.action_min = np.asarray(action_min_droid, dtype=np.float32)
        self.action_max = np.asarray(action_max_droid, dtype=np.float32)
        self.action_denominator = self.action_max - self.action_min
        self.action_denominator = np.where(self.action_denominator == 0, 1.0, self.action_denominator)

        self._episodes = []
        self._data_files = {}
        self._num_records_per_epoch = 0
        self._epoch = 0
        self._episode_cache = OrderedDict()
        self._data_file_cache = OrderedDict()
        self._task_index_to_text = {}
        self._episode_index_to_instruction = {}
        self._data_template = None
        self._video_template = None

        self._load_metadata()

    def _load_metadata(self):
        info_path = os.path.join(self.data_path, "meta", "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Droid LeRobot info.json not found: {info_path}")
        with open(info_path, "r") as f:
            info = json.load(f)

        self.fps = float(info.get("fps", 15.0) if self.fps is None else self.fps)
        self._data_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        self._video_template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

        self._episode_index_to_instruction = self._load_annotated_tasks()
        self._task_index_to_text = self._load_standard_tasks()

        episode_paths = sorted(glob.glob(os.path.join(self.data_path, "meta", "episodes", "chunk-*", "file-*.parquet")))
        if not episode_paths:
            raise FileNotFoundError(f"No Droid episode metadata parquet files found under {self.data_path}/meta/episodes")

        episode_columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
        for key in VIDEO_KEYS:
            episode_columns.extend([
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
            ])

        frames = [pd.read_parquet(path, columns=episode_columns) for path in episode_paths]
        episodes_df = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        if self.length is not None:
            episodes_df = episodes_df.iloc[: int(self.length)].reset_index(drop=True)

        file_starts = (
            episodes_df.groupby(["data/chunk_index", "data/file_index"], sort=False)["dataset_from_index"]
            .min()
            .to_dict()
        )

        verified_data_paths = set()
        verified_video_paths = set()

        for _, row in episodes_df.iterrows():
            episode_index = int(row["episode_index"])
            data_chunk = int(row["data/chunk_index"])
            data_file = int(row["data/file_index"])
            file_id = (data_chunk, data_file)
            data_path = self._get_data_file_path(file_id)

            traj_len = int(row["length"])
            dataset_from = int(row["dataset_from_index"])
            file_frame_offset = dataset_from - int(file_starts[file_id])
            if file_frame_offset < 0:
                raise ValueError(f"Negative local frame offset for episode {episode_index}.")

            videos = {}
            frame_caches = {}
            video_offsets = {}
            for video_key in VIDEO_KEYS:
                video_chunk = int(row.get(f"videos/{video_key}/chunk_index", data_chunk))
                video_file = int(row.get(f"videos/{video_key}/file_index", data_file))
                videos[video_key] = _format_lerobot_path(
                    self.data_path,
                    self._video_template,
                    chunk_index=video_chunk,
                    file_index=video_file,
                    video_key=video_key,
                )
                frame_caches[video_key] = _format_frame_cache_path(
                    self.data_path,
                    chunk_index=video_chunk,
                    file_index=video_file,
                    video_key=video_key,
                    image_resize_size=self.image_resize_size,
                )

                timestamp = row.get(f"videos/{video_key}/from_timestamp", np.nan)
                if pd.notna(timestamp):
                    video_offsets[video_key] = int(round(float(timestamp) * self.fps))
                else:
                    video_offsets[video_key] = file_frame_offset

            if data_path not in verified_data_paths:
                if not os.path.isfile(data_path):
                    raise FileNotFoundError(f"Droid LeRobot parquet not found: {data_path}")
                verified_data_paths.add(data_path)
            for key in EXTERIOR_VIDEO_KEYS:
                video_path = videos[key]
                if video_path in verified_video_paths:
                    continue
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Droid LeRobot video not found: {video_path}")
                verified_video_paths.add(video_path)

            instruction = self._episode_index_to_instruction.get(episode_index, "")
            if not _is_valid_text(instruction):
                instruction = _first_task(row.get("tasks", ""))

            self._episodes.append(
                {
                    "episode_index": episode_index,
                    "length": traj_len,
                    "valid_length": self._get_valid_length(traj_len),
                    "instruction": str(instruction).strip(),
                    "data_file_id": file_id,
                    "data_frame_offset": int(file_frame_offset),
                    "videos": videos,
                    "frame_caches": frame_caches,
                    "video_offsets": video_offsets,
                }
            )
            self._num_records_per_epoch += self._get_episode_sample_count(traj_len)

        if self._num_records_per_epoch <= 0:
            raise RuntimeError(f"No Droid training frames found under: {self.data_path}")

    def _load_annotated_tasks(self):
        path = os.path.join(self.data_path, "meta", "tasks_annotated.parquet")
        if not os.path.isfile(path):
            return {}
        df = pd.read_parquet(path, columns=["task"])
        mapping = {}
        for episode_index, task in df["task"].items():
            if _is_valid_text(task):
                mapping[int(episode_index)] = str(task).strip()
        return mapping

    def _load_standard_tasks(self):
        path = os.path.join(self.data_path, "meta", "tasks.parquet")
        if not os.path.isfile(path):
            return {}
        df = pd.read_parquet(path)
        if "task_index" not in df.columns:
            return {}
        return {
            int(task_index): str(task_text)
            for task_text, task_index in df["task_index"].items()
            if _is_valid_text(task_text)
        }

    def _get_data_file_path(self, file_id):
        file_id = (int(file_id[0]), int(file_id[1]))
        path = self._data_files.get(file_id)
        if path is not None:
            return path
        path = _format_lerobot_path(
            self.data_path,
            self._data_template,
            chunk_index=file_id[0],
            file_index=file_id[1],
        )
        self._data_files[file_id] = path
        return path

    def _get_valid_length(self, traj_len):
        traj_len = int(traj_len)
        if self.allow_end_padding:
            return traj_len
        if self.load_future_video:
            return max(0, traj_len - self.future_len)
        if self.load_future_image and self.future_image_mode == "horizon":
            return max(0, traj_len - self.future_len)
        return max(0, traj_len - self.future_len + 1)

    def _get_episode_sample_count(self, traj_len):
        valid_len = self._get_valid_length(traj_len)
        if valid_len <= 0:
            return 0
        if self.full_sequence:
            return valid_len
        return min(valid_len, max(1, int(valid_len * self.sampling_rate)))

    def _load_data_file(self, file_id):
        cached = self._data_file_cache.get(file_id)
        if cached is not None:
            self._data_file_cache.move_to_end(file_id)
            return cached

        data_path = self._get_data_file_path(file_id)
        df = pd.read_parquet(data_path, columns=DATA_COLUMNS)
        self._data_file_cache[file_id] = df
        self._data_file_cache.move_to_end(file_id)
        while len(self._data_file_cache) > self.data_file_cache_size:
            self._data_file_cache.popitem(last=False)
        return df

    def _read_episode_arrays(self, episode_id):
        episode = self._episodes[episode_id]
        traj_len = int(episode["length"])
        offset = int(episode["data_frame_offset"])
        df = self._load_data_file(episode["data_file_id"])

        episode_df = df.iloc[offset: offset + traj_len]
        if len(episode_df) != traj_len or int(episode_df["episode_index"].iloc[0]) != int(episode["episode_index"]):
            episode_df = df[df["episode_index"] == int(episode["episode_index"])]
        if "frame_index" in episode_df.columns:
            frame_indices = episode_df["frame_index"].to_numpy()
            if len(frame_indices) > 1 and np.any(np.diff(frame_indices) < 0):
                episode_df = episode_df.sort_values("frame_index").reset_index(drop=True)

        raw_actions = _stack_column(episode_df[ACTION_KEY], width=7)
        delta_pose = raw_actions[:, :6]
        delta_pose = 2.0 * (delta_pose - self.action_min) / self.action_denominator - 1.0
        delta_pose = np.clip(delta_pose, -1.0, 1.0)
        gripper_action = 2.0 * np.clip(raw_actions[:, 6:7], 0.0, 1.0) - 1.0
        actions_np = np.concatenate([delta_pose, gripper_action], axis=1).astype(np.float32)

        state_pose = _stack_column(episode_df[STATE_POSE_KEY], width=6)
        gripper_state = _stack_column(episode_df[STATE_GRIPPER_KEY], width=1)
        gripper_state = np.clip(gripper_state, 0.0, 1.0)
        proprio_np = np.concatenate([state_pose, gripper_state], axis=1).astype(np.float32)

        lap_language_actions_np = None
        if self.load_language_action and self.language_action_format == "lap":
            movement = compute_padded_movement_actions(state_pose)
            gripper_open = _binarize_open_gripper_from_close_position(raw_actions[:, 6:7], threshold=0.5)
            lap_language_actions_np = np.concatenate([movement, gripper_open], axis=1).astype(np.float32)

        instruction = episode["instruction"]
        if not _is_valid_text(instruction):
            if "task_index" in episode_df.columns:
                task_index = int(episode_df["task_index"].iloc[0])
                instruction = self._task_index_to_text.get(task_index, "")
            if not _is_valid_text(instruction) and "language_instruction" in episode_df.columns:
                instruction = str(episode_df["language_instruction"].iloc[0])

        arrays = {
            "actions": actions_np,
            "proprio": proprio_np,
            "instruction": str(instruction).strip(),
        }
        if lap_language_actions_np is not None:
            arrays["lap_language_actions"] = lap_language_actions_np
        return arrays

    def _load_episode_arrays(self, episode_id):
        cached = self._episode_cache.get(episode_id)
        if cached is not None:
            self._episode_cache.move_to_end(episode_id)
            return cached

        arrays = self._read_episode_arrays(episode_id)
        self._episode_cache[episode_id] = arrays
        self._episode_cache.move_to_end(episode_id)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return arrays

    def _get_shard_records(self):
        epoch = self._epoch
        self._epoch += 1
        records = self._build_epoch_records(epoch)
        rank, world_size, worker_id, num_workers = self._get_shard_info()
        return self._shard_records(records, rank, world_size, worker_id, num_workers)

    def _get_shard_info(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        return rank, world_size, worker_id, num_workers

    @staticmethod
    def _shard_records(records, rank, world_size, worker_id, num_workers):
        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        return records[shard_index::total_shards]

    def _build_epoch_records(self, epoch):
        rng = np.random.default_rng(self.seed + int(epoch))
        records_by_file = defaultdict(list)
        for episode_id, episode in enumerate(self._episodes):
            valid_len = int(episode["valid_length"])
            if valid_len <= 0:
                continue

            if self.full_sequence:
                frame_indices = np.arange(valid_len, dtype=np.int64)
            else:
                sample_count = min(valid_len, max(1, int(valid_len * self.sampling_rate)))
                frame_indices = rng.choice(valid_len, size=sample_count, replace=False)

            view_choices = rng.integers(0, len(EXTERIOR_VIDEO_KEYS), size=len(frame_indices))
            file_records = records_by_file[episode["data_file_id"]]
            for frame_index, view_choice in zip(frame_indices, view_choices):
                file_records.append((episode_id, int(frame_index), EXTERIOR_VIDEO_KEYS[int(view_choice)]))

        file_ids = list(records_by_file.keys())
        if file_ids:
            rng.shuffle(file_ids)

        records = []
        for file_id in file_ids:
            file_records = records_by_file[file_id]
            if file_records:
                order = rng.permutation(len(file_records))
                records.extend(file_records[int(i)] for i in order)
        return records

    def _build_batch(self, records):
        grouped = defaultdict(list)
        for output_index, (episode_id, frame_index, main_video_key) in enumerate(records):
            grouped[episode_id].append((output_index, frame_index, main_video_key))

        samples = [None] * len(records)
        for episode_id, positions in grouped.items():
            episode = self._episodes[episode_id]
            arrays = self._load_episode_arrays(episode_id)
            actions_np = arrays["actions"]
            proprio_np = arrays["proprio"]
            lap_language_actions_np = arrays.get("lap_language_actions")
            traj_len = actions_np.shape[0]

            image_indices_by_key = defaultdict(list)
            wrist_image_indices = []
            for _, t, main_video_key in positions:
                t = min(int(t), traj_len - 1)
                indices = self._get_observation_image_indices(t, traj_len)
                image_indices_by_key[main_video_key].extend(indices)
                if self.view_mode == "multi":
                    wrist_image_indices.extend(indices)

            if self.load_future_image:
                for _, t, main_video_key in positions:
                    target_idx = self._get_future_image_index(min(int(t), traj_len - 1), traj_len)
                    image_indices_by_key[main_video_key].append(target_idx)
                    if self.view_mode == "multi":
                        wrist_image_indices.append(target_idx)

            if self.load_future_video:
                for _, t, main_video_key in positions:
                    future_indices = self._get_future_video_indices(min(int(t), traj_len - 1), traj_len).tolist()
                    image_indices_by_key[main_video_key].extend(future_indices)
                    if self.view_mode == "multi":
                        wrist_image_indices.extend(future_indices)

            main_frames_by_key = {
                key: self._read_episode_frames(episode, key, indices)
                for key, indices in image_indices_by_key.items()
            }
            wrist_frames = None
            if self.view_mode == "multi":
                wrist_frames = self._read_episode_frames(episode, WRIST_VIDEO_KEY, wrist_image_indices)

            pad_action_hist = actions_np[0].astype(np.float32) if self.emit_history_actions else None
            pad_action_fut = actions_np[-1].astype(np.float32)
            for output_index, t, main_video_key in positions:
                t = min(int(t), traj_len - 1)
                main_frames = main_frames_by_key[main_video_key]

                need_hist_obs = self.input_modality == "video" or self.emit_proprioception
                if need_hist_obs:
                    hist_indices_obs = np.arange(t - self.history_len + 1, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)
                else:
                    hist_indices_obs = None

                fut_indices = np.arange(t, t + self.future_len)
                fut_acts_np = np.tile(pad_action_fut, (self.future_len, 1)).astype(np.float32)
                valid_mask_fut = fut_indices < traj_len
                if np.any(valid_mask_fut):
                    fut_acts_np[valid_mask_fut] = actions_np[fut_indices[valid_mask_fut]]

                sample = {
                    "future_actions": torch.from_numpy(fut_acts_np),
                    "instruction": arrays["instruction"],
                }

                if self.load_language_action:
                    if self.language_action_format == "vla-0":
                        sample["language_action_text"] = format_language_action_text(
                            fut_acts_np,
                            language_action_format=self.language_action_format,
                            num_bins=self.language_action_num_bins,
                        )
                    elif self.language_action_format == "lap":
                        if lap_language_actions_np is None:
                            raise KeyError("Missing `lap_language_actions` for Droid LAP language-action.")
                        valid_len = max(1, min(self.future_len, traj_len - int(t)))
                        sample["language_action_text"] = format_language_action_text(
                            lap_language_actions_np[int(t): int(t) + valid_len],
                            language_action_format=self.language_action_format,
                            num_bins=self.language_action_num_bins,
                        )

                if self.load_future_image:
                    target_idx = self._get_future_image_index(t, traj_len)
                    sample["future_image"] = main_frames[target_idx].copy()
                    if self.view_mode == "multi":
                        sample["future_image_wrist"] = wrist_frames[target_idx].copy()

                if self.load_future_video:
                    future_video_indices = self._get_future_video_indices(t, traj_len)
                    future_video = np.stack([main_frames[int(i)] for i in future_video_indices], axis=0)
                    if self.view_mode == "multi":
                        future_video_wrist = np.stack([wrist_frames[int(i)] for i in future_video_indices], axis=0)
                        future_video = np.concatenate([future_video, future_video_wrist], axis=2)
                    sample["future_video"] = future_video.copy()

                if self.emit_proprioception:
                    sample["proprioception"] = torch.from_numpy(proprio_np[hist_indices_obs])

                if self.emit_history_actions:
                    hist_indices_act = np.arange(t - self.history_len, t)
                    hist_actions = np.tile(pad_action_hist, (self.history_len, 1)).astype(np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = np.clip(hist_indices_act[valid_mask], 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    sample["history_actions"] = torch.from_numpy(hist_actions)

                if self.input_modality == "video":
                    sample["video"] = np.stack([main_frames[int(i)] for i in hist_indices_obs], axis=0)
                    if self.view_mode == "multi":
                        sample["video_wrist"] = np.stack([wrist_frames[int(i)] for i in hist_indices_obs], axis=0)
                elif self.input_modality == "image":
                    sample["image"] = main_frames[t].copy()
                    if self.view_mode == "multi":
                        sample["image_wrist"] = wrist_frames[t].copy()
                else:
                    raise ValueError(f"Unknown input_modality: {self.input_modality}")

                samples[output_index] = sample

        return samples

    def _get_observation_image_indices(self, t, traj_len):
        if self.input_modality == "video":
            indices = np.arange(t - self.history_len + 1, t + 1)
            return np.clip(indices, 0, traj_len - 1).tolist()
        return [min(int(t), traj_len - 1)]

    def _read_episode_frames(self, episode, video_key, frame_indices):
        rel_indices = sorted({int(i) for i in frame_indices})
        if not rel_indices:
            return {}

        video_offset = int(episode["video_offsets"].get(video_key, episode["data_frame_offset"]))
        file_indices = [video_offset + idx for idx in rel_indices]

        cache_path = episode.get("frame_caches", {}).get(video_key)
        file_frames = _read_frame_cache_frames(cache_path, file_indices)
        if file_frames is None:
            video_path = episode["videos"][video_key]
            file_frames = _read_video_frames(video_path, file_indices)
            if self.image_resize_size is not None:
                file_frames = {
                    idx: resize_frame_uint8(frame, self.image_resize_size)
                    for idx, frame in file_frames.items()
                }

        return {
            rel_idx: file_frames[file_idx]
            for rel_idx, file_idx in zip(rel_indices, file_indices)
        }

    def _get_future_image_index(self, t, traj_len):
        if self.future_image_mode == "goal":
            return traj_len - 1
        if self.future_image_mode == "horizon":
            return min(int(t) + self.future_len, traj_len - 1)
        raise ValueError(f"Unknown future_image_mode: {self.future_image_mode}")

    def _get_future_video_indices(self, t, traj_len):
        offsets = np.arange(0, self.future_len + 1, self.future_video_downsample, dtype=np.int64)
        return np.clip(int(t) + offsets, 0, traj_len - 1)

    def __iter__(self):
        shard_records = self._get_shard_records()
        for start in range(0, len(shard_records), self.batch_size):
            batch_records = shard_records[start:start + self.batch_size]
            if len(batch_records) < self.batch_size and self.drop_last:
                break
            yield self._build_batch(batch_records)

    def __len__(self):
        if self.drop_last:
            return self._num_records_per_epoch // self.batch_size
        return int(np.ceil(self._num_records_per_epoch / self.batch_size))


def _iter_data_parquet_paths(root):
    return sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))


def _compute_action_min_max(root):
    action_min = np.full(6, np.inf, dtype=np.float64)
    action_max = np.full(6, -np.inf, dtype=np.float64)
    total_rows = 0

    for path in _iter_data_parquet_paths(root):
        df = pd.read_parquet(path, columns=[ACTION_KEY])
        actions = _stack_column(df[ACTION_KEY], width=7)[:, :6]
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        total_rows += actions.shape[0]
        print(f"[ok] {path} rows={actions.shape[0]}")

    if total_rows == 0:
        raise RuntimeError(f"No Droid data parquet files found under {root}")
    return action_min.astype(np.float32), action_max.astype(np.float32), total_rows


def main():
    parser = argparse.ArgumentParser(description="Compute Droid action.original[:6] min/max stats.")
    parser.add_argument(
        "root",
        nargs="?",
        default="/data/NTU_slab/draven/data/MolmoAct2-DROID",
        help="Path to the Droid LeRobot dataset root.",
    )
    args = parser.parse_args()

    action_min, action_max, total_rows = _compute_action_min_max(args.root)
    print(f"Scanned {total_rows} frames")
    print(f"action_min_droid = {action_min.tolist()}")
    print(f"action_max_droid = {action_max.tolist()}")


if __name__ == "__main__":
    main()
