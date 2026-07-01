import json
import os
from collections import OrderedDict, defaultdict

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from src.datasets.libero_act import (
    get_libero_action_stats,
    normalize_image_resize_size,
    resize_frame_uint8,
)
from src.datasets.language_action import (
    build_libero_lap_language_actions,
    format_language_action_text,
    normalize_language_action_format,
)


MAIN_VIDEO_KEY = "observation.images.image"
WRIST_VIDEO_KEY = "observation.images.wrist_image"
FRAME_CACHE_DIR = "frame_cache"


def _read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_lerobot_path(root, template, episode_chunk, episode_index, video_key=None):
    kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return os.path.join(root, template.format(**kwargs))


def _format_frame_cache_path(root, episode_chunk, episode_index, video_key, image_resize_size):
    if image_resize_size is None:
        return None
    h, w = image_resize_size
    return os.path.join(
        root,
        FRAME_CACHE_DIR,
        f"resize_{h}x{w}",
        f"chunk-{episode_chunk:03d}",
        video_key,
        f"episode_{episode_index:06d}.npy",
    )


def _to_uint8(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    return frame.astype(np.uint8)


def _convert_open01_gripper_to_libero(gripper_action):
    """Convert LeRobot open_gripper 1=open/0=close to LIBERO -1=open/+1=close."""
    gripper_action = np.asarray(gripper_action, dtype=np.float32)
    return 1.0 - 2.0 * np.clip(gripper_action, 0.0, 1.0)


def _read_video_frames(video_path, frame_indices):
    """Read selected RGB frames from a LeRobot episode mp4."""
    unique_indices = sorted({int(i) for i in frame_indices})
    if not unique_indices:
        return {}

    if len(unique_indices) == 1:
        idx = unique_indices[0]
        return {idx: _to_uint8(iio.imread(video_path, index=idx))}

    needed = set(unique_indices)
    max_idx = unique_indices[-1]
    frames = {}
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx in needed:
            frames[idx] = _to_uint8(frame)
            if len(frames) == len(needed):
                break
        if idx >= max_idx:
            break

    missing = needed.difference(frames)
    if missing:
        raise IndexError(f"Missing frames {sorted(missing)} while reading {video_path}")
    return frames


def _read_frame_cache_frames(cache_path, frame_indices):
    unique_indices = sorted({int(i) for i in frame_indices})
    if not unique_indices:
        return {}
    if not cache_path or not os.path.isfile(cache_path):
        return None

    frames_np = np.load(cache_path, mmap_mode="r")
    return {
        idx: np.asarray(frames_np[idx], dtype=np.uint8).copy()
        for idx in unique_indices
    }


class _LiberoLeRobotBase(IterableDataset):
    def __init__(
        self,
        suite_paths,
        dataset_name="libero",
        normalization_suite_name=None,
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
        seed=0,
        image_resize_size=None,
        load_language_action=False,
        language_action_format="vla-0",
        language_action_num_bins=1000,
        action_mode="libero",
    ):
        super().__init__()
        self.suite_paths = suite_paths
        self.dataset_name = dataset_name
        self.normalization_suite_name = normalization_suite_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.load_future_video = bool(load_future_video)
        self.future_video_downsample = int(future_video_downsample)
        if self.future_video_downsample <= 0:
            raise ValueError(f"future_video_downsample must be positive, got {future_video_downsample}.")
        if self.load_future_video and self.future_len % self.future_video_downsample != 0:
            raise ValueError("future_len must be divisible by future_video_downsample.")
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.allow_end_padding = allow_end_padding
        self.emit_proprioception = emit_proprioception
        self.emit_history_actions = emit_history_actions
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.episode_cache_size = int(episode_cache_size)
        self.seed = int(seed)
        self.image_resize_size = normalize_image_resize_size(image_resize_size)
        self.load_language_action = bool(load_language_action)
        self.language_action_format = normalize_language_action_format(language_action_format)
        self.language_action_num_bins = int(language_action_num_bins)
        self.action_mode = str(action_mode or "libero").lower()
        self.action_dim = None

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.input_modality not in {"image", "video"}:
            raise ValueError(f"Unknown input_modality: {self.input_modality}")
        if self.view_mode not in {"single", "multi"}:
            raise ValueError(f"Unknown view_mode: {self.view_mode}")
        if self.action_mode not in {"libero", "latent"}:
            raise ValueError("action_mode must be one of: libero, latent")
        if self.load_language_action and self.action_mode == "latent":
            raise ValueError("language-action labels are only supported for LIBERO 7D actions.")

        self._episodes = []
        self._num_records_per_epoch = 0
        self._epoch = 0
        self._task_maps = {}
        self._episode_cache = OrderedDict()
        self._episode_arrays_preloaded = False
        self._load_metadata()
        self._init_action_padding()
        self._preload_episode_arrays()

    def _action_dim_from_info(self, info):
        action_feature = info.get("features", {}).get("action", {})
        shape = action_feature.get("shape", None)
        if isinstance(shape, list) and len(shape) == 1:
            return int(shape[0])
        return None

    def _validate_and_set_action_dim(self, info, suite_root):
        action_feature = info.get("features", {}).get("action", {})
        action_info = action_feature.get("info", {}) or {}
        metadata_is_latent = bool(action_info.get("latent_action", False))
        action_dim = self._action_dim_from_info(info)

        if self.action_mode == "libero":
            if metadata_is_latent:
                raise ValueError(
                    f"Dataset at {suite_root} contains latent actions; set data.action_mode='latent'."
                )
            if action_dim is not None and action_dim != 7:
                raise ValueError(
                    f"LIBERO action mode expects 7D actions, got action shape [{action_dim}] in {suite_root}."
                )
            self.action_dim = 7
            return

        if action_dim is None:
            raise ValueError(f"Cannot infer latent action dim from metadata: {suite_root}")
        if self.action_dim is None:
            self.action_dim = action_dim
        elif self.action_dim != action_dim:
            raise ValueError(
                f"All latent-action suites must share one action dim; "
                f"got {action_dim} for {suite_root}, expected {self.action_dim}."
            )

    def _init_action_padding(self):
        if self.load_language_action and self.action_mode != "libero":
            raise ValueError("language-action labels are only supported for LIBERO 7D actions.")
        if self.action_mode == "libero":
            self.action_min, self.action_max = get_libero_action_stats(self.normalization_suite_name)
            self.action_denominator = self.action_max - self.action_min
            self.action_denominator = np.where(self.action_denominator == 0, 1.0, self.action_denominator)

            zero_pose = np.zeros(6, dtype=np.float32)
            self.normalized_zero_pose = 2.0 * (zero_pose - self.action_min) / self.action_denominator - 1.0
            self.normalized_zero_pose = np.clip(self.normalized_zero_pose, -1.0, 1.0)
            self.history_action_pad = np.concatenate(
                [self.normalized_zero_pose, np.array([-1.0], dtype=np.float32)]
            )
        else:
            if self.action_dim is None:
                raise ValueError("Latent action dim was not resolved from metadata.")
            self.action_min = None
            self.action_max = None
            self.action_denominator = None
            self.normalized_zero_pose = None
            self.history_action_pad = np.zeros(self.action_dim, dtype=np.float32)

    # for each episode, load metadata and compute the number of valid samples for training.
    def _load_metadata(self):
        for suite_name, suite_root in self.suite_paths:
            info_path = os.path.join(suite_root, "meta", "info.json")
            episodes_path = os.path.join(suite_root, "meta", "episodes.jsonl")
            tasks_path = os.path.join(suite_root, "meta", "tasks.jsonl")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
            if not os.path.isfile(episodes_path):
                raise FileNotFoundError(f"LeRobot episodes.jsonl not found: {episodes_path}")

            with open(info_path, "r") as f:
                info = json.load(f)
            self._validate_and_set_action_dim(info, suite_root)

            data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
            video_template = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            chunks_size = int(info.get("chunks_size", 1000))
            episodes = _read_jsonl(episodes_path)
            if os.path.isfile(tasks_path):
                self._task_maps[suite_name] = {
                    int(row["task_index"]): row["task"]
                    for row in _read_jsonl(tasks_path)
                }
            else:
                self._task_maps[suite_name] = {}
            if self.length is not None:
                episodes = episodes[: int(self.length)]

            for episode in episodes:
                episode_index = int(episode["episode_index"])
                episode_chunk = episode_index // chunks_size
                traj_len = int(episode["length"])
                instruction = ""
                if episode.get("tasks"):
                    instruction = episode["tasks"][0]

                parquet_path = _format_lerobot_path(
                    suite_root,
                    data_template,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                main_video_path = _format_lerobot_path(
                    suite_root,
                    video_template,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=MAIN_VIDEO_KEY,
                )
                wrist_video_path = _format_lerobot_path(
                    suite_root,
                    video_template,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=WRIST_VIDEO_KEY,
                )
                main_frame_cache_path = _format_frame_cache_path(
                    suite_root,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=MAIN_VIDEO_KEY,
                    image_resize_size=self.image_resize_size,
                )
                wrist_frame_cache_path = _format_frame_cache_path(
                    suite_root,
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=WRIST_VIDEO_KEY,
                    image_resize_size=self.image_resize_size,
                )
                if not os.path.isfile(parquet_path):
                    raise FileNotFoundError(f"LeRobot parquet not found: {parquet_path}")
                if not os.path.isfile(main_video_path):
                    raise FileNotFoundError(f"LeRobot video not found: {main_video_path}")

                episode_id = len(self._episodes)
                self._episodes.append(
                    {
                        "suite_name": suite_name,
                        "episode_index": episode_index,
                        "length": traj_len,
                        "valid_length": self._get_valid_length(traj_len),
                        "instruction": instruction,
                        "parquet_path": parquet_path,
                        "videos": {
                            MAIN_VIDEO_KEY: main_video_path,
                            WRIST_VIDEO_KEY: wrist_video_path,
                        },
                        "frame_caches": {
                            MAIN_VIDEO_KEY: main_frame_cache_path,
                            WRIST_VIDEO_KEY: wrist_frame_cache_path,
                        },
                    }
                )

                self._num_records_per_epoch += self._get_episode_sample_count(traj_len)

        if self._num_records_per_epoch <= 0:
            roots = ", ".join(path for _, path in self.suite_paths)
            raise RuntimeError(f"No LeRobot training frames found under: {roots}")

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
        return min(valid_len, max(1, int(valid_len * float(self.sampling_rate))))

    def _read_episode_arrays(self, episode_id):
        episode = self._episodes[episode_id]
        df = pd.read_parquet(episode["parquet_path"])
        if "frame_index" in df.columns:
            frame_indices = df["frame_index"].to_numpy()
            if len(frame_indices) > 1 and np.any(np.diff(frame_indices) < 0):
                df = df.sort_values("frame_index").reset_index(drop=True)

        raw_actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions[:, None]
        raw_actions_libero = None
        if self.action_mode == "libero":
            raw_actions_libero = raw_actions.copy()
            raw_actions_libero[:, 6:7] = _convert_open01_gripper_to_libero(raw_actions[:, 6:7])

            delta_pose = raw_actions_libero[:, :6]
            delta_pose = 2.0 * (delta_pose - self.action_min) / self.action_denominator - 1.0
            delta_pose = np.clip(delta_pose, -1.0, 1.0)
            gripper_action = np.clip(raw_actions_libero[:, 6:7], -1.0, 1.0)
            actions_np = np.concatenate([delta_pose, gripper_action], axis=1)
        else:
            actions_np = raw_actions

        raw_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        gripper_qpos = raw_state[:, 6:8]
        gripper_state = 1.0 - (np.mean(np.abs(gripper_qpos), axis=1, keepdims=True) / 0.04)
        gripper_state = np.clip(gripper_state, 0.0, 1.0)
        proprio_np = np.concatenate([raw_state[:, :6], gripper_state], axis=1)
        lap_language_actions_np = None
        if self.load_language_action and self.language_action_format == "lap":
            if raw_actions_libero is None:
                raise ValueError("LAP language actions require LIBERO 7D actions.")
            lap_language_actions_np = build_libero_lap_language_actions(raw_state, raw_actions_libero)

        instruction = episode["instruction"]
        if not instruction and "task_index" in df.columns:
            task_index = int(df["task_index"].iloc[0])
            instruction = self._task_maps.get(episode["suite_name"], {}).get(task_index, str(task_index))

        arrays = {
            "actions": actions_np,
            "proprio": proprio_np,
            "instruction": instruction,
        }
        if lap_language_actions_np is not None:
            arrays["lap_language_actions"] = lap_language_actions_np
        return arrays

    def _preload_episode_arrays(self):
        for episode_id in range(len(self._episodes)):
            self._episode_cache[episode_id] = self._read_episode_arrays(episode_id)
        self._episode_arrays_preloaded = True

    def _load_episode_arrays(self, episode_id):
        cached = self._episode_cache.get(episode_id)
        if cached is not None:
            if not self._episode_arrays_preloaded:
                self._episode_cache.move_to_end(episode_id)
            return cached

        arrays = self._read_episode_arrays(episode_id)
        self._episode_cache[episode_id] = arrays
        self._episode_cache.move_to_end(episode_id)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return arrays

    # for each epoch, build the list of (episode_id, frame_index) records for all valid frames across episodes, then shard them according to distributed and dataloader worker info.
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
        records = []
        for episode_id, episode in enumerate(self._episodes):
            valid_len = int(episode["valid_length"])
            if valid_len <= 0:
                continue

            if self.full_sequence:
                frame_indices = np.arange(valid_len, dtype=np.int64)
            else:
                sample_count = min(valid_len, max(1, int(valid_len * float(self.sampling_rate))))
                frame_indices = rng.choice(valid_len, size=sample_count, replace=False)

            records.extend((episode_id, int(frame_index)) for frame_index in frame_indices)

        if records:
            order = rng.permutation(len(records))
            records = [records[int(i)] for i in order]
        return records

    # for a batch of (episode_id, frame_index) records, load the corresponding data and build the batch dict.
    def _build_batch(self, records):
        grouped = defaultdict(list)
        for output_index, (episode_id, frame_index) in enumerate(records):
            grouped[episode_id].append((output_index, frame_index))

        samples = [None] * len(records)
        for episode_id, positions in grouped.items():
            episode = self._episodes[episode_id]
            arrays = self._load_episode_arrays(episode_id)
            actions_np = arrays["actions"]
            proprio_np = arrays["proprio"]
            lap_language_actions_np = arrays.get("lap_language_actions")
            traj_len = actions_np.shape[0]

            image_indices = []
            if self.input_modality == "video":
                for _, t in positions:
                    t = min(int(t), traj_len - 1)
                    hist_indices_obs = np.arange(t - self.history_len + 1, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)
                    image_indices.extend(hist_indices_obs.tolist())
            else:
                image_indices = [min(int(t), traj_len - 1) for _, t in positions]
            if self.load_future_image:
                image_indices.extend(
                    self._get_future_image_index(min(int(t), traj_len - 1), traj_len)
                    for _, t in positions
                )
            if self.load_future_video:
                for _, t in positions:
                    image_indices.extend(self._get_future_video_indices(min(int(t), traj_len - 1), traj_len).tolist())

            main_frames = self._read_episode_frames(episode, MAIN_VIDEO_KEY, image_indices)
            wrist_frames = None
            if self.view_mode == "multi":
                wrist_frames = self._read_episode_frames(episode, WRIST_VIDEO_KEY, image_indices)

            pad_action_hist = self.history_action_pad if self.emit_history_actions else None
            pad_action_fut = actions_np[-1].astype(np.float32)

            for output_index, t in positions:
                t = min(int(t), traj_len - 1)

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
                            raise KeyError("Missing `lap_language_actions` for LeRobot LAP language-action.")
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

                samples[output_index] = sample

        return samples
    
    def _read_episode_frames(self, episode, video_key, frame_indices):
        cache_path = episode.get("frame_caches", {}).get(video_key)
        cached_frames = _read_frame_cache_frames(cache_path, frame_indices)
        if cached_frames is not None:
            return cached_frames

        video_path = episode["videos"][video_key]
        frames = _read_video_frames(video_path, frame_indices)
        if self.image_resize_size is None:
            return frames
        return {
            idx: resize_frame_uint8(frame, self.image_resize_size)
            for idx, frame in frames.items()
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

    # iterate over the epoch records in batches.
    def __iter__(self):
        shard_records = self._get_shard_records()
        for start in range(0, len(shard_records), self.batch_size):
            batch_records = shard_records[start:start + self.batch_size]
            if len(batch_records) < self.batch_size and self.drop_last:
                break
            yield self._build_batch(batch_records)

    # return the number of batches per epoch.
    def __len__(self):
        if self.drop_last:
            return self._num_records_per_epoch // self.batch_size
        return int(np.ceil(self._num_records_per_epoch / self.batch_size))


class LiberoLeRobotAct(_LiberoLeRobotBase):
    def __init__(
        self,
        data_path,
        dataset_name="libero",
        normalization_suite_name=None,
        **kwargs,
    ):
        super().__init__(
            suite_paths=[(dataset_name, data_path)],
            dataset_name=dataset_name,
            normalization_suite_name=normalization_suite_name,
            **kwargs,
        )


class LiberoMixedLeRobotAct(_LiberoLeRobotBase):
    """LeRobot LIBERO mixed loader with one global random frame index pool."""

    SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

    def __init__(
        self,
        data_root,
        dataset_name="libero_mixed",
        normalization_suite_name="libero_mixed",
        **kwargs,
    ):
        super().__init__(
            suite_paths=[(suite, os.path.join(data_root, suite)) for suite in self.SUITES],
            dataset_name=dataset_name,
            normalization_suite_name=normalization_suite_name,
            **kwargs,
        )
