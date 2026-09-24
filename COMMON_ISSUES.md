# VLANeXt Common Issues

Below are frequently encountered issues and their solutions. Feel free to open an issue if your problem is not listed here.

---

### 1. Evaluation Environment Error

**Symptom:** Errors related to missing shared libraries when running LIBERO or LIBERO-plus evaluation.

**Solution:** Install the required system packages:

```bash
apt-get update
apt-get install -y \
  libgl1-mesa-glx libegl1-mesa \
  libxrandr2 libxcursor1 libxinerama1 libxrender1 \
  libgl1-mesa-dev libegl1-mesa-dev
```

---

### 2. Evaluation GPU and MuJoCo Error

**Symptom:** Evaluation works on GPU 0 but fails on other GPUs. For example:

```bash
# ✅ Works
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval

# ❌ Fails
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 python -m scripts.libero_bench_eval
```

**Solution:** List the target GPU first followed by GPU 0, and set `MUJOCO_EGL_DEVICE_ID=0`:

```bash
CUDA_VISIBLE_DEVICES=1,0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval
```

> **Why this works:** `CUDA_VISIBLE_DEVICES=1,0` remaps physical GPU 1 to logical device 0. Setting `MUJOCO_EGL_DEVICE_ID=0` then correctly targets the remapped device. And we set CUDA_VISIBLE_DEVICES has device 0 to avoid the GPU check in MuJoCo.


### 3. Numpy Error during LIBERO-plus Evaluation

**Symptom:** `np.float_` was removed in the NumPy 2.0 release error when evaluation.

**Solution:** Change `np.float_` to `np.float64` in `./third_party/LIBERO-plus/libero/libero/envs/env_wrapper.py`:

```python
# file: ./third_party/LIBERO-plus/libero/libero/envs/env_wrapper.py
# line 105
# Change np.float_ to np.float64
```


### 4. Checkpoint Unexpected when Using LLaMA Family

**Symptom:** Warning about unexpected keys when loading a checkpoint with a LLaMA-family VLM, e.g., `text_model.xx` keys not being loaded.

**Solution:** This is expected and **not** an error. Since we only use the visual part of SigLip, the text part will not be loaded. All `text_model.*` keys in the checkpoint are safely ignored.



### 5. Flash Attention crashes with illegal memory access on Qwen3.5

**Symptom:** Flash Attention crashes with illegal memory access on Qwen3.5.

**Solution:** Replace the `_is_packed_sequence function` in `transformers/modeling_flash_attention_utils.py` with the following function, which can handle the 3D positional embedding.

```python
def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False
    
    # Extract the temporal dimension to support multi-dimensional RoPE
    t_position_ids = position_ids[0] if position_ids.dim() > 2 else position_ids
    
    return batch_size == 1 and (t_position_ids[:, 1:] - t_position_ids[:, :-1] < 0).sum().bool()
```


### 6. FileNotFoundError: [Errno 2] No such file or directory: '' when evaluation

**Symptom:** Encounter the bug FileNotFoundError: [Errno 2] No such file or directory: '' when evaluation

**Solution:** Run `unset LIBERO_CONFIG_PATH` before evaluation will fix the problem.


### 7. Import Error when using Flash Attention

**Symptom:** Encounter `ImportError: xxx undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`

**Solution:** This is the flash-attention version problem. Just `pip install flash-attn==2.7.4.post1 --no-build-isolation`.