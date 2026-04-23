import os
import gc
import ctypes
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], 'GPU')

# Force glibc to return freed memory to the OS (Linux only)
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _LIBC.malloc_trim(0)
except Exception:
    def _malloc_trim():
        pass

class DroidAct(IterableDataset):
    def __init__(
        self,
        droid_path,
        dataset_name='droid',
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=False,
        input_modality="video",
        view_mode="single",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=30000,
    ):
        super().__init__()
        self.droid_path = droid_path
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.buffer_size = buffer_size

    def __iter__(self):
        builder = tfds.builder_from_directory(builder_dir=self.droid_path)
        
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        droid_ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)
        
        if self.length is not None:
            droid_ds = droid_ds.take(self.length)

        shuffle_buffer = []
        BUFFER_SIZE = self.buffer_size
        
        cam_key = 'exterior_image_1_left'
        wrist_key = 'wrist_image_left'

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

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        ds_iterator = droid_ds.shard(num_shards=total_shards, index=shard_index)

        ds_iter = iter(ds_iterator)
        traj_id = -1
        while True:
            try:
                try:
                    traj_data = next(ds_iter)
                except StopIteration:
                    break
                except tf.errors.DataLossError as e:
                    traj_id += 1
                    print(f"[Warn] Skipping trajectory {traj_id}: TF DataLossError during iteration: {e}")
                    continue
                traj_id += 1
                
                traj_batch = next(iter(traj_data['steps'].batch(5000)))

                if traj_batch['reward'][-1].numpy() != 1:
                    del traj_batch
                    continue

                traj_len = traj_batch['action'].shape[0]
                
                images_np = traj_batch['observation'][cam_key].numpy()
                if images_np.dtype != np.uint8:
                    images_np = (images_np * 255).astype(np.uint8)

                wrist_np = None
                if self.view_mode == "multi":
                    obs = traj_batch['observation']
                    if wrist_key in obs:
                        wrist_np = obs[wrist_key].numpy()
                        if wrist_np.dtype != np.uint8:
                            wrist_np = (wrist_np * 255).astype(np.uint8)
                    else:
                        wrist_np = images_np.copy()
                    del obs

                # Process Proprioception: Cartesian + Gripper
                cart_pos = traj_batch['observation']['cartesian_position']
                
                # [Proprio Gripper]
                # Raw: 1D position in [0, 1]. 0 = Open, 1 = Closed.
                # Processed: Direct copy.
                # Result: Range [0, 1]. 0 = Open, 1 = Closed.
                gripper_pos = traj_batch['observation']['gripper_position']
                proprio_np = tf.concat([cart_pos, gripper_pos], axis=-1).numpy().astype(np.float32)

                # Process Actions: Cartesian Velocity + Gripper Command (Normalized)
                # DROID Actions: 6 (vel) + 1 (gripper)
                cart_vel = traj_batch['action_dict']['cartesian_velocity']
                cart_vel = tf.cast(cart_vel, tf.float32)
                cart_vel = tf.clip_by_value(cart_vel, -1.0, 1.0)
                
                # [Action Gripper]
                # Raw: 1D position in [0, 1]. 0 = Open, 1 = Closed.
                # Processed: Binarized to {-1, 1} based on 0.5 threshold.
                # Result: Range [-1, 1]. -1 = Open, 1 = Closed.
                grip_pos = traj_batch['action_dict']['gripper_position']
                grip_cmd = tf.where(grip_pos > 0.5, 1.0, -1.0) 
                grip_cmd = tf.cast(grip_cmd, tf.float32)

                actions_np = tf.concat([cart_vel, grip_cmd], axis=-1).numpy().astype(np.float32)
                
                instruction = traj_batch['language_instruction'][0].numpy().decode('utf-8')

                del traj_batch, cart_pos, gripper_pos, cart_vel, grip_pos, grip_cmd

                if self.full_sequence:
                    sample_indices = np.arange(traj_len)
                else:
                    num_samples = max(1, int(traj_len / (15 * 5)))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)
                    
                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)
                    
                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    hist_imgs = images_np[hist_indices_obs]
                    hist_imgs_wrist = wrist_np[hist_indices_obs] if wrist_np is not None else None
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])
                    
                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = hist_indices_act[valid_mask]
                        valid_indices = np.clip(valid_indices, 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)
                    
                    fut_acts_np = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        valid_indices_fut = fut_indices[valid_mask_fut]
                        fut_acts_np[valid_mask_fut] = actions_np[valid_indices_fut]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    sample = {
                        'proprioception': hist_proprio,
                        'history_actions': hist_actions,
                        'future_actions': fut_acts,
                        'instruction': instruction,
                    }

                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + self.future_len, traj_len - 1)
                        sample['future_image'] = images_np[target_idx].copy()

                    if self.input_modality == "video":
                        sample['video'] = hist_imgs
                        if self.view_mode == "multi":
                            sample['video_wrist'] = hist_imgs_wrist
                    elif self.input_modality == "image":
                        sample['image'] = images_np[t].copy()
                        if self.view_mode == "multi":
                            sample['image_wrist'] = wrist_np[t].copy() if wrist_np is not None else images_np[t].copy()
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)
                    
                    if len(shuffle_buffer) >= BUFFER_SIZE:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

                del images_np, actions_np, proprio_np
                if wrist_np is not None:
                    del wrist_np

            except Exception as e:
                print(f"[Warn] Skipping trajectory {traj_id} due to error: {e}")
                continue
            finally:
                if traj_id % 50 == 0:
                    gc.collect()
                    _malloc_trim()
        
        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    """
    Fast stats: count how many training samples DroidAct would yield.
    Now includes a progress bar (tqdm).
    """
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_path", type=str, default="/mnt/NTU_slab/draven/data/open_x_embodiment/droid/1.0.1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit_traj", type=int, default=None)
    parser.add_argument("--full_sequence", action="store_true")
    args = parser.parse_args()

    builder = tfds.builder_from_directory(builder_dir=args.droid_path)
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split=args.split, shuffle_files=False, read_config=read_config)

    total_files = builder.info.splits[args.split].num_examples
    if args.limit_traj is not None:
        ds = ds.take(int(args.limit_traj))
        total_files = min(total_files, int(args.limit_traj))

    total_trajs = 0
    success_trajs = 0
    total_samples = 0
    SUBSAMPLE_DENOM = 15 * 5

    print(f"Scanning {total_files} trajectories from {args.droid_path}...")
    pbar = tqdm(enumerate(ds), total=total_files, unit="traj", desc="Scanning")

    for traj_id, traj_data in pbar:
        total_trajs += 1
        try:
            # Load only necessary data
            traj_batch = next(iter(traj_data["steps"].batch(5000)))

            if traj_batch["reward"][-1].numpy() != 1:
                continue

            success_trajs += 1
            traj_len = int(traj_batch["action"].shape[0])

            if args.full_sequence:
                total_samples += traj_len
            else:
                total_samples += max(1, int(traj_len / SUBSAMPLE_DENOM))

            pbar.set_postfix({"Succ": success_trajs, "Samples": total_samples})

        except Exception as e:
            pbar.write(f"[Warn] Skipping traj {traj_id}: {e}")
            continue

    print("\n" + "="*40)
    print(f"DONE. Split: {args.split} | FullSeq: {args.full_sequence}")
    print(f"Total Trajectories: {total_trajs}")
    print(f"Successful Trajs:   {success_trajs}")
    print(f"Total Samples:      {total_samples}")
    print("="*40)