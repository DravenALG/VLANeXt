import os
import glob
import h5py
import numpy as np
import torch
import cv2
from torch.utils.data import IterableDataset

# Initialize action statistics based on dataset_name
action_min_stack_the_yellow_block_on_the_green_block = [-0.03949321433901787, -0.002825927920639515, -1.9244569540023804, -0.6661514639854431, 0.10408835113048553, -2.0748939514160156, 0.0, -0.9651939868927002, 0.004326112102717161, -2.205113410949707, -0.9689792990684509, 0.030561888590455055, -0.2881574332714081, -0.1]
action_max_stack_the_yellow_block_on_the_green_block = [0.9352949261665344, 2.212195634841919, -0.0005407640128396451, 1.1022863388061523, 1.2159340381622314, 0.6330427527427673, 0.1, 0.05055271089076996, 2.3829026222229004, 0.004360999912023544, 0.28662237524986267, 1.2197368144989014, 1.4918632507324219, 0.01]

class RealWorldAct(IterableDataset):
    def __init__(
        self,
        data_path,
        bimanual=False,
        history_len=15,
        future_len=15,
        full_sequence=True,
        input_modality="video",
        view_mode="single",
        buffer_size=5000,
        sampling_rate=0.1,
        episode_downsample_factor=1,
        allow_end_padding=True,
        load_future_image=False,
        future_image_mode="horizon",
        dataset_name="real_world",
        seed=0,
    ):
        super().__init__()
        self.data_path = data_path
        self.bimanual = bimanual
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.episode_downsample_factor = int(episode_downsample_factor)
        if self.episode_downsample_factor < 1:
            raise ValueError(
                f"episode_downsample_factor must be >= 1, got {episode_downsample_factor}"
            )
        self.allow_end_padding = allow_end_padding
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.dataset_name = dataset_name  # Used as instruction
        self.seed = int(seed)
        self._epoch = 0

        # Determine action stats from dataset name (all data is aloha format)
        if "stack_the_yellow_block_on_the_green_block" in dataset_name:
            if bimanual:
                self.action_min = np.array(action_min_stack_the_yellow_block_on_the_green_block)
                self.action_max = np.array(action_max_stack_the_yellow_block_on_the_green_block)
            else:
                # Right arm only: indices 7:14
                self.action_min = np.array(action_min_stack_the_yellow_block_on_the_green_block[7:14])
                self.action_max = np.array(action_max_stack_the_yellow_block_on_the_green_block[7:14])
        else:
            raise ValueError(f"Unknown dataset name '{dataset_name}'")

    def _process_image(self, img, target_size):
        # Resize: (W, H)
        img = cv2.resize(img, target_size)
        # Assuming Data is RGB.
        return img

    def _get_future_image_index(self, t, traj_len):
        if self.future_image_mode in {"last", "goal"}:
            return traj_len - 1
        if self.future_image_mode == "horizon":
            return min(int(t) + self.future_len, traj_len - 1)
        raise ValueError(f"Unknown future_image_mode: {self.future_image_mode}")

    def _process_aloha_trajectory(self, file_path):
        imgs_wrist2 = None  # Initialize default
        with h5py.File(file_path, 'r') as f:
            if "observations/qpos" not in f or "action" not in f: return None

            # Read qpos (Current Joint State / Proprioception) - (T, 14)
            qpos = f["observations/qpos"][:]

            # Read actions directly from dataset - (T, 14)
            actions = f["action"][:]

            imgs_high = f["observations/images/cam_high"][:]

            # Select arm data based on bimanual flag
            if self.bimanual:
                imgs_wrist = f["observations/images/cam_left_wrist"][:]
                imgs_wrist2 = f["observations/images/cam_right_wrist"][:]
                proprio_seq = qpos
                action_seq = actions
            else:
                # Default to right arm for single-arm aloha
                imgs_wrist = f["observations/images/cam_right_wrist"][:]
                proprio_seq = qpos[:, 7:14]
                action_seq = actions[:, 7:14]

            traj_len = min(len(proprio_seq), len(action_seq), len(imgs_high), len(imgs_wrist))
            if imgs_wrist2 is not None:
                traj_len = min(traj_len, len(imgs_wrist2))

            proprio_seq = proprio_seq[:traj_len]
            action_seq = action_seq[:traj_len]
            imgs_high = imgs_high[:traj_len]
            imgs_wrist = imgs_wrist[:traj_len]
            if imgs_wrist2 is not None:
                imgs_wrist2 = imgs_wrist2[:traj_len]

            if self.episode_downsample_factor > 1:
                stride = self.episode_downsample_factor
                proprio_seq = proprio_seq[::stride]
                action_seq = action_seq[::stride]
                imgs_high = imgs_high[::stride]
                imgs_wrist = imgs_wrist[::stride]
                if imgs_wrist2 is not None:
                    imgs_wrist2 = imgs_wrist2[::stride]

            # Resize Images (480x640 -> 240x320)
            target_size = (320, 240)
            images_main = np.array([self._process_image(im, target_size) for im in imgs_high])
            images_wrist = np.array([self._process_image(im, target_size) for im in imgs_wrist])
            images_wrist2 = (
                np.array([self._process_image(im, target_size) for im in imgs_wrist2])
                if imgs_wrist2 is not None else None
            )

            return proprio_seq, action_seq, images_main, images_wrist, images_wrist2

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

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1

        traj_files = sorted(glob.glob(os.path.join(self.data_path, "*.hdf5")))
        rank, world_size, worker_id, num_workers = self._get_shard_info()
        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        traj_files = traj_files[shard_index::total_shards]

        rng_seed = self.seed + epoch * 100003 + shard_index * 1009
        rng = np.random.default_rng(rng_seed)

        shuffle_buffer = []

        for traj_path in traj_files:
            try:
                data = self._process_aloha_trajectory(traj_path)

                if data is None: continue
                proprio_seq, action_seq, images_main, images_wrist, images_wrist2 = data

                traj_len = len(proprio_seq)
                if traj_len < 2: continue

                # Normalize Data (MinMax only)
                epsilon = 1e-6
                denominator = self.action_max - self.action_min
                denominator = np.where(denominator < epsilon, 1.0, denominator)

                # Normalize Proprioception (Current Joints)
                norm_proprio = 2.0 * (proprio_seq - self.action_min) / denominator - 1.0
                norm_proprio = np.clip(norm_proprio, -1.0, 1.0).astype(np.float32)

                # Normalize Actions (Target Joints)
                norm_actions = 2.0 * (action_seq - self.action_min) / denominator - 1.0
                norm_actions = np.clip(norm_actions, -1.0, 1.0).astype(np.float32)

                # Edge-padding values: first/last action (zero is invalid for joint positions)
                pad_action_hist = norm_actions[0]   # shape (D,)
                pad_action_fut  = norm_actions[-1]  # shape (D,)

                instruction = self.dataset_name

                # --- Sample indices ---
                if self.full_sequence:
                    sample_indices = np.arange(traj_len)
                else:
                    num_samples = max(1, int(traj_len * self.sampling_rate))
                    sample_indices = rng.choice(traj_len, size=num_samples, replace=False)

                if not self.allow_end_padding:
                    sample_indices = sample_indices[sample_indices + self.future_len <= traj_len]

                for t in sample_indices:
                    # 1. History
                    hist_indices_obs = np.clip(
                        np.arange(t - self.history_len + 1, t + 1), 0, traj_len - 1
                    )

                    # 2. Future
                    fut_indices = np.arange(t, t + self.future_len)

                    # Gather Images
                    hist_imgs = images_main[hist_indices_obs]
                    hist_imgs_wrist = images_wrist[hist_indices_obs]
                    hist_imgs_wrist2 = images_wrist2[hist_indices_obs] if images_wrist2 is not None else None

                    # History Proprioception
                    hist_proprio = torch.from_numpy(norm_proprio[hist_indices_obs])

                    # History Actions
                    hist_act_np = np.tile(pad_action_hist, (self.history_len, 1))
                    hist_act_indices = np.arange(t - self.history_len, t)
                    valid_hist = hist_act_indices >= 0
                    if np.any(valid_hist):
                        hist_act_np[valid_hist] = norm_actions[np.clip(hist_act_indices[valid_hist], 0, traj_len - 1)]
                    hist_actions = torch.from_numpy(hist_act_np)

                    # Future Actions — pad with last action instead of zeros
                    f_acts_np = np.tile(pad_action_fut, (self.future_len, 1))
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        f_acts_np[valid_mask_fut] = norm_actions[fut_indices[valid_mask_fut]]
                    fut_actions = torch.from_numpy(f_acts_np)

                    sample = {
                        'proprioception': hist_proprio,
                        'history_actions': hist_actions,
                        'future_actions': fut_actions,
                        'instruction': instruction,
                    }

                    if self.load_future_image:
                        target_idx = self._get_future_image_index(t, traj_len)
                        sample['future_image'] = images_main[target_idx].copy()

                    # Payload
                    if self.input_modality == "video":
                        sample['video'] = hist_imgs
                        if self.view_mode == "multi":
                            sample['video_wrist'] = hist_imgs_wrist
                            if hist_imgs_wrist2 is not None:
                                sample['video_wrist2'] = hist_imgs_wrist2
                    elif self.input_modality == "image":
                        sample['image'] = images_main[t]
                        if self.view_mode == "multi":
                            sample['image_wrist'] = images_wrist[t]
                            if images_wrist2 is not None:
                                sample['image_wrist2'] = images_wrist2[t]

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        idx = int(rng.integers(len(shuffle_buffer)))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

            except Exception as e:
                print(f"Error processing {traj_path}: {e}")
                continue

        # Yield remaining
        rng.shuffle(shuffle_buffer)
        for s in shuffle_buffer:
            yield s

if __name__ == "__main__":
    """
    Compute action statistics (min/max) for RealWorldAct dataset.
    Updated for Direct Joint Values.
    """
    from tqdm import tqdm

    # Configuration
    DATA_ROOT = "/data/NTU_slab/draven/data/real_world_data/cobot_magic_data/stack_the_yellow_block_on_the_green_block/"  # Update this path
    ARMS_TYPE = "bimanual"                         # 'single', 'bimanual'

    print(f"Scanning RealWorld dataset in {DATA_ROOT} [aloha, {ARMS_TYPE}]...")

    traj_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.hdf5")))

    # Initialize min/max
    dim_action = 14 if ARMS_TYPE == 'bimanual' else 7

    act_min = np.full(dim_action, np.inf)
    act_max = np.full(dim_action, -np.inf)

    total_trajs = 0
    total_samples = 0

    pbar = tqdm(traj_files)
    for traj_path in pbar:
        try:
            action_seq = None

            # --- Extract Actions (Joints) ---
            with h5py.File(traj_path, 'r') as f:
                if "action" not in f: continue

                # Read actions directly from dataset
                full_actions = f["action"][:]

                if ARMS_TYPE == 'single':
                    # Right arm only
                    action_seq = full_actions[:, 7:14]
                elif ARMS_TYPE == 'bimanual':
                    action_seq = full_actions
                else:
                    raise ValueError(f"Unknown ARMS_TYPE '{ARMS_TYPE}'")

            if action_seq is None or len(action_seq) < 1:
                continue

            # Update Global Stats
            current_min = np.min(action_seq, axis=0)
            current_max = np.max(action_seq, axis=0)
            act_min = np.minimum(act_min, current_min)
            act_max = np.maximum(act_max, current_max)

            total_trajs += 1
            total_samples += len(action_seq)
            pbar.set_postfix({"Trajs": total_trajs})

        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue

    print(f"\n--- RealWorld Stats ({ARMS_TYPE}) ---")
    print(f"Total Trajectories: {total_trajs}")
    print(f"Total Samples: {total_samples}")
    print(f"action_min = {act_min.tolist()}")
    print(f"action_max = {act_max.tolist()}")
