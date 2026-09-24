<p align="center">
  <img src="imgs/LOGO.png" alt="VLANeXt logo" width="80">
</p>

# VLANeXt Family: From Core Recipes to Emerging Paradigms

<!-- RELEASE TODO: 2609.00000 is a fictitious arXiv placeholder for VLANeXt Family. Replace every occurrence in this README when the identifier is assigned. -->
[![VLANeXt Paper](https://img.shields.io/badge/VLANeXt-2602.18532-b31b1b.svg)](https://arxiv.org/abs/2602.18532)
[![VLANeXt Family Paper](https://img.shields.io/badge/VLANeXt_Family-2609.00000-b31b1b.svg)](https://arxiv.org/abs/2609.00000)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://dravenalg.github.io/projects/VLANeXt/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/DravenALG/VLANeXt)
[![Awesome VLA & WAM](https://img.shields.io/badge/GitHub-AwesomeVLA&WAM-black)](https://github.com/DravenALG/awesome-vla-wam)

**A simple and research-oriented codebase for studying Vision-Language-Action models.** This repository provides the official implementation of two connected works:

| Paper | Scope |
| --- | --- |
| **[VLANeXt: Recipes for Building Strong VLA Models](https://arxiv.org/abs/2602.18532)**<br>ICML 2026 | A systematic study of the VLA design space, distilling 12 findings into the core VLANeXt recipe. |
| **[VLANeXt Family: A Systematic Study of VLA Models from Core Recipes to Emerging Paradigms](https://arxiv.org/abs/2609.00000)**<br>arXiv 2026 · Extended version | Demonstrates the robustness and generality of the core recipe across model scales and emerging VLA paradigms. |

VLANeXt Family builds directly on our ICML work, with the original VLANeXt serving as the base model. **Both papers share one codebase**, with a common model implementation, training entry point, and evaluation pipeline. Variants are selected through configuration files.

[Models and configurations](#models-and-configurations) · [Environment setup](#environment-setup) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

## Changelog & News

- **2026.09** 🚀 **VLANeXt Family is released!** Five new variants, **S, L, LAM, JEPA, and WAM**, extend the core recipe across model scales and emerging paradigms within one shared framework.
- **2026.06** 🎉 **VLANeXt** has been accepted to **ICML 2026**!

## 📖 From VLANeXt to the VLANeXt Family

**The core recipe.** Our ICML study systematically examines **foundational components, perception essentials, and action modeling perspectives** under a unified setup. Starting from a simple RT-2-style baseline, we conduct more than 500 experiments and distill 12 findings into a practical recipe. The resulting VLANeXt achieves **97.4% on LIBERO** and **83.9% on LIBERO-plus**, averaged across four suites, with further validation in real-world manipulation.

**A robust recipe across paradigms.** VLANeXt Family extends this study to four emerging directions. Controlled ablations show that **the core recipe remains effective across these settings**, demonstrating its robustness and generality as VLA paradigms evolve:

- **Model scaling (S / Base / L):** apply the recipe to compact and larger pretrained backbones.
- **Latent-action pretraining (LAM):** learn from visual transitions before fine-tuning on labeled robot actions.
- **Latent predictive representation learning (JEPA):** jointly learn actions and predict future semantic features in DINOv3 space.
- **World action modeling (WAM):** jointly learn future visual dynamics and robot actions with a pretrained video-generation backbone.

<p align="center">
  <img src="imgs/roadmap.png" alt="VLANeXt roadmap: from the core design study to model scaling, LAM, JEPA, and WAM" width="600">
</p>

<a id="models-and-configurations"></a>

## 🧩 Models and Configurations

All variants use [the shared training script](scripts/train.py) and [the VLANeXt model](src/models/VLANeXt.py). Select a YAML below to configure the backbone and learning objectives.

| Model | Backbone | Focus | LIBERO avg. (%) | Training configuration |
| --- | --- | --- | ---: | --- |
| **VLANeXt (Base / B)** | Qwen3-VL-2B | Core recipe from ICML; base model of the family | 97.4 | [VLANeXt-B](config/VLANeXt-B_config.yaml) |
| **VLANeXt-S** | Qwen3.5-0.8B | Compact VLA | 96.7 | [VLANeXt-S](config/VLANeXt-S_config.yaml) |
| **VLANeXt-L** | Qwen3-VL-4B | Larger VLA | **98.4** | [VLANeXt-L](config/VLANeXt-L_config.yaml) |
| **VLANeXt-LAM** | Qwen3-VL-2B | Latent-action pretraining (VQ-VAE) | 97.6 | [VLANeXt-LAM](config/VLANeXt-LAM_config.yaml) |
| **VLANeXt-JEPA** | Qwen3-VL-2B | Future DINOv3 feature prediction | 97.7 | [VLANeXt-JEPA](config/VLANeXt-JEPA_config.yaml) |
| **VLANeXt-WAM** | Wan2.2-TI2V-5B | Joint video and action learning (fast connection) | 98.2 | [VLANeXt-WAM](config/VLANeXt-WAM_config.yaml) |

Results are success rates reported in the papers, averaged over Spatial, Object, Goal, and Long.

Pass the desired YAML to `--config`. [libero_train_config.yaml](config/libero_train_config.yaml) uses the Base settings. The LAM preset is for latent-action pretraining; complete the fine-tuning stage below before evaluation.

The main controls for the extensions are:

| Direction | Configuration controls |
| --- | --- |
| Model scaling | `model.lmm_path` and `model.policy_depth`, as set in the S / B / L presets. |
| Latent-action pretraining | `data.action_mode: latent`, with `model.action_dim` matching the LAM's `model.latent_dim`. |
| JEPA-style prediction | `model.future_image_loss_weight: 1.0`, `model.future_image_prediction_type: dinov3_flow`, and `model.future_image_mode: horizon`. |
| World action modeling | The WAM preset uses Wan with tight conditioning, flow matching, and `model.video_generation_loss_weight: 1.0`; `model.wan_action_condition_mode` selects `fast` or `joint`. For WAM, set `data.augmentation.enabled: false` during training and `data.augmentation.center_crop: false` during evaluation. |


<a id="environment-setup"></a>

## 🛠️ Environment Setup

### Basic Installation

Run the commands below from the repository root. Before training, update `data.data_root`, `project.output_dir`, and any local model or checkpoint paths in the selected YAML. Give each experiment a distinct `project.name` to keep its checkpoints separate.

```bash
# Basic setup
conda create -n codebase python=3.10
conda activate codebase
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
conda install -c conda-forge ffmpeg
```

### Benchmark Installation

**LIBERO**

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
pip install ./third_party/LIBERO
```

**LIBERO-plus**

Create a separate environment, e.g. `codebase-plus`, using the basic installation steps above, then install LIBERO-plus in that environment:

```bash
mkdir -p third_party
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
pip install ./third_party/LIBERO-plus
# Dependencies
apt install libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
pip install -r third_party/LIBERO-plus/extra_requirements.txt
conda env config vars set LIBERO_CONFIG_PATH=~/.libero_plus
```

Download the benchmark assets following [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus), then reactivate the environment so `LIBERO_CONFIG_PATH` takes effect.

<a id="training"></a>

## 🚀 Training

### Prepare LIBERO Data

The shared data pipeline supports **TFDS/RLDS** and **LeRobot** formats. Choose the format and set `data.dataset_format` and `data.data_root` accordingly.

**TFDS/RLDS:** use the modified LIBERO dataset provided by [OpenVLA](https://github.com/openvla/openvla), with `data.dataset_format: tfds`.

```bash
hf download openvla/modified_libero_rlds --repo-type dataset --local-dir datasets/LIBERO_modified
```

**LeRobot:** use the dataset provided by [FastWAM](https://github.com/yuantianyuan01/FastWAM), with `data.dataset_format: lerobot`. The Family presets use this format by default.

```bash
hf download yuanty/LIBERO-fastwam --repo-type dataset --local-dir datasets/LIBERO_fastwam

# Cache decoded frames to speed up training
python src/datasets/build_libero_lerobot_frame_cache.py datasets/LIBERO_fastwam --resize-size 256
```

### Train a VLANeXt Variant

Use the same command for Base, S, L, JEPA, and WAM by changing the configuration path. For LAM, first follow the latent-action workflow below.

```bash
# Single GPU: set train.distributed: false and train.deepspeed.enabled: false
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/VLANeXt-B_config.yaml

# Multi-GPU: set train.distributed: true; enable DeepSpeed in the YAML if desired
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m scripts.train --config config/VLANeXt-B_config.yaml
```

`data.batch_size` is the total batch size across GPUs. Edit the YAML directly; `--config` selects a complete configuration file.

### Latent-Action Pretraining and Fine-Tuning

VLANeXt-LAM uses a latent action model to derive supervision from visual transitions, then transfers the pretrained VLA to real robot actions.

**1. Train the latent action model.** Set the source dataset and output paths in [libero_train_lam_config.yaml](config/libero_train_lam_config.yaml). Choose `model.lam_type` from `vq` (VQ-VAE, the default), `vae`, or `vicreg`.

```bash
# Single GPU: set train.distributed: false in the LAM configuration
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_lam --config config/libero_train_lam_config.yaml

# Multi-GPU: set train.distributed: true
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m scripts.train_lam --config config/libero_train_lam_config.yaml
```

**2. Generate latent-action data.** Use the trained LAM checkpoint to create a LeRobot dataset copy whose `action` column stores latent actions. Replace the checkpoint path with the output of step 1.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.generate_lam \
  --checkpoint /path/to/lam/checkpoint_final.pt \
  --source-root datasets/LIBERO_fastwam \
  --output-root datasets/LIBERO_fastwam_lam
```

**3. Pretrain VLANeXt on latent actions.** In [VLANeXt-LAM_config.yaml](config/VLANeXt-LAM_config.yaml), set `data.data_root` to the generated dataset, keep `data.action_mode: latent`, and match `model.action_dim` to the LAM's `model.latent_dim` (7 by default). Use a new `project.name` for this stage.

```bash
# Uses the distributed settings in the preset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m scripts.train --config config/VLANeXt-LAM_config.yaml
```

**4. Fine-tune on real actions.** Use [the Base configuration](config/VLANeXt-B_config.yaml) with the original LIBERO dataset, `data.action_mode: libero`, and `model.action_dim: 7`. Set `train.pretrained_checkpoint` to the **VLA checkpoint from step 3**. If the action dimensions differ, set `train.pretrained_ignore_mismatched_shapes: true`. Train under a new `project.name` and evaluate the fine-tuned checkpoint.

### FAST Tokenizer Training

For experiments with FAST action tokenization, train the tokenizer with:

```bash
python -m scripts.train_FAST --config config/libero_train_fast_config.yaml
```

Then set `model.loss_type: classification`, `model.fast_action_tokenizer.enabled: true`, and `model.fast_action_tokenizer.tokenizer_path` to the tokenizer output directory in your VLA training YAML. Train with the shared `scripts.train` entry point.

### DROID Training

The codebase also supports [DROID](https://droid-dataset.github.io). We use the reorganized and filtered LeRobot dataset released by [MolmoAct2](https://github.com/allenai/molmoact2). Download it and update `data.data_root` in [droid_train_config.yaml](config/droid_train_config.yaml):

```bash
hf download allenai/MolmoAct2-DROID-Dataset --repo-type dataset --local-dir datasets/MolmoAct2-DROID

# Single GPU: set train.distributed: false and train.deepspeed.enabled: false
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config config/droid_train_config.yaml

# Multi-GPU: set train.distributed: true; enable DeepSpeed in the YAML if desired
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m scripts.train --config config/droid_train_config.yaml
```

<a id="evaluation"></a>

## 📊 Evaluation

Both papers and all real-action variants use the same evaluation entry points. The model architecture and training configuration are loaded from the checkpoint.

In [libero_bench_config.yaml](config/libero_bench_config.yaml) or [libero_plus_bench_config.yaml](config/libero_plus_bench_config.yaml), set:

- `eval.finetuned_checkpoint`: the trained `.pt` file or DeepSpeed checkpoint directory.
- `eval.task_suite_name`: `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` (Long).
- `eval.num_parallel_envs`: the number of parallel environments; use `1` for sequential evaluation.
- `eval.num_trials_per_task` and `model.diffusion_steps`: the evaluation budget and inference steps for the intended experiment.

### LIBERO Benchmark

Activate the LIBERO environment and run from the repository root. See the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) for benchmark details.

```bash
export PYTHONPATH="${PWD}/third_party/LIBERO"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_bench_eval --config config/libero_bench_config.yaml
```

### LIBERO-plus Benchmark

Activate the separate LIBERO-plus environment and run from the repository root. Models are trained on LIBERO and evaluated under the unseen perturbations in [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus).

```bash
export PYTHONPATH="${PWD}/third_party/LIBERO-plus"

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m scripts.libero_plus_bench_eval --config config/libero_plus_bench_config.yaml
```

## ❗ Common Issues

If you run into issues, check [COMMON_ISSUES.md](COMMON_ISSUES.md) for known problems and solutions.

<a id="citation"></a>

## 📚 Citation

If you build on the core VLANeXt recipe, please cite our ICML paper. If you use the Family variants or their extended studies, please also cite VLANeXt Family.

```bibtex
@inproceedings{wu2026vlanext,
  title={VLANeXt: Recipes for Building Strong VLA Models},
  author={Xiao-Ming Wu and Bin Fan and Kang Liao and Jian-jian Jiang and Runze Yang and Yihang Luo and Zhonghua Wu and Wei-Shi Zheng and Chen Change Loy},
  booktitle={ICML},
  year={2026}
}

@article{wu2026vlanextfamily,
  title={VLANeXt Family: A Systematic Study of VLA Models from Core Recipes to Emerging Paradigms},
  author={Xiao-Ming Wu and Kang Liao and Yihang Luo and Bin Fan and Jian-jian Jiang and Runze Yang and Zhonghua Wu and Wei-Shi Zheng and Chen Change Loy},
  journal={arXiv preprint arXiv:2609.00000},
  year={2026}
}
```

## 🗞️ License

This project is licensed under [NTU S-Lab License 1.0](LICENSE).
