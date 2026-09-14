import sys
from pathlib import Path

import torch
import torch.nn as nn


class WanProcessorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class WanBackbone(nn.Module):
    """Thin wrapper around the local Wan2.2 TI2V components."""

    def __init__(
        self,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="google/umt5-xxl",
        text_len=512,
        torch_dtype=torch.bfloat16,
        device="cpu",
    ):
        super().__init__()
        self.model_id = model_id
        self.tokenizer_model_id = tokenizer_model_id
        self.text_len = int(text_len)
        self.torch_dtype = torch_dtype

        wan_root = Path(__file__).resolve().parent / "Wan2.2"
        if str(wan_root) not in sys.path:
            sys.path.insert(0, str(wan_root))

        from huggingface_hub import snapshot_download
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE

        checkpoint_dir = Path(snapshot_download(model_id))
        tokenizer_dir = checkpoint_dir / "google" / "umt5-xxl"
        if not tokenizer_dir.exists():
            tokenizer_snapshot = Path(snapshot_download(tokenizer_model_id))
            nested_tokenizer_dir = tokenizer_snapshot / "google" / "umt5-xxl"
            tokenizer_dir = nested_tokenizer_dir if nested_tokenizer_dir.exists() else tokenizer_snapshot

        self.dit = WanModel.from_pretrained(str(checkpoint_dir))
        self.dit.to(device=device, dtype=torch_dtype)

        text_encoder = T5EncoderModel(
            text_len=self.text_len,
            dtype=torch_dtype,
            device=torch.device(device),
            checkpoint_path=str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(tokenizer_dir),
        )
        self.text_encoder = text_encoder
        self.text_encoder_model = text_encoder.model
        self.tokenizer = text_encoder.tokenizer

        vae = Wan2_2_VAE(
            vae_pth=str(checkpoint_dir / "Wan2.2_VAE.pth"),
            dtype=torch_dtype,
            device=device,
        )
        self.vae = vae
        self.vae_model = vae.model
        self.processor = WanProcessorWrapper(self.tokenizer)
        self._sync_nonmodule_components()

    @property
    def dtype(self):
        return self.torch_dtype

    @property
    def device(self):
        return next(self.dit.parameters()).device

    @property
    def hidden_size(self):
        return int(self.dit.dim)

    @property
    def latent_channels(self):
        return int(getattr(self.vae, "z_dim", self.dit.in_dim))

    @property
    def temporal_downsample_factor(self):
        return 4

    @property
    def spatial_downsample_factor(self):
        return 16

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dtype = kwargs.get("dtype", None)
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            self.torch_dtype = dtype
        self._sync_nonmodule_components()
        return self

    def _apply(self, fn):
        result = super()._apply(fn)
        self._sync_nonmodule_components()
        return result

    def train(self, mode=True):
        super().train(mode)
        self.vae_model.eval()
        self.text_encoder_model.eval()
        return self

    def _sync_nonmodule_components(self):
        device = self.device
        self.text_encoder.device = device
        self.text_encoder.dtype = self.torch_dtype
        self.text_encoder.model = self.text_encoder_model
        self.vae.device = device
        self.vae.dtype = self.torch_dtype
        self.vae.model = self.vae_model
        self.vae.scale = [
            scale.to(device=device, dtype=self.torch_dtype)
            for scale in self.vae.scale
        ]

    @torch.no_grad()
    def encode_videos(self, videos):
        self._sync_nonmodule_components()
        if videos.ndim != 5:
            raise ValueError(f"`videos` must be [B, C, T, H, W], got {tuple(videos.shape)}")
        self.vae_model.eval()
        videos_list = [
            video.to(device=self.device, dtype=self.torch_dtype)
            for video in videos
        ]
        latents = self.vae.encode(videos_list)
        if latents is None:
            raise RuntimeError("WAN VAE failed to encode videos.")
        return torch.stack(latents, dim=0).to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def decode_latents(self, latents):
        self._sync_nonmodule_components()
        self.vae_model.eval()
        latents_list = [
            latent.to(device=self.device, dtype=self.torch_dtype)
            for latent in latents
        ]
        videos = self.vae.decode(latents_list)
        if videos is None:
            raise RuntimeError("WAN VAE failed to decode latents.")
        return torch.stack(videos, dim=0)

    def encode_prompts(self, prompt_texts, extra_context=None):
        self._sync_nonmodule_components()
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
        ids, mask = self.tokenizer(
            prompt_texts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        self.text_encoder_model.eval()
        with torch.no_grad():
            context = self.text_encoder_model(ids, mask)

        seq_lens = mask.gt(0).sum(dim=1).long()
        context_list = []
        for i, seq_len in enumerate(seq_lens.tolist()):
            text_ctx = context[i, : int(seq_len)].to(dtype=self.torch_dtype)
            if extra_context is not None:
                extra = extra_context[i].to(device=self.device, dtype=self.torch_dtype)
                max_text = max(1, self.text_len - extra.shape[0])
                text_ctx = text_ctx[:max_text]
                text_ctx = torch.cat([text_ctx, extra], dim=0)
            context_list.append(text_ctx)
        return context_list

    def forward_latents(
        self,
        latents,
        timesteps,
        context,
        *,
        return_hidden_states=False,
        self_attn_mask=None,
    ):
        self._sync_nonmodule_components()
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B, C, T, H, W], got {tuple(latents.shape)}")
        if isinstance(context, torch.Tensor):
            context = [u for u in context]
        patch_t, patch_h, patch_w = self.dit.patch_size
        _, _, frames, height, width = latents.shape
        seq_len = (frames // patch_t) * (height // patch_h) * (width // patch_w)
        seq_len = int(seq_len)

        hidden_states = []
        handles = []
        if return_hidden_states:
            def _hook(_module, _inputs, output):
                hidden_states.append(output.to(dtype=self.torch_dtype))

            handles = [block.register_forward_hook(_hook) for block in self.dit.blocks]

        autocast_enabled = self.device.type == "cuda" and self.torch_dtype in (torch.float16, torch.bfloat16)
        try:
            with torch.autocast(device_type="cuda", dtype=self.torch_dtype, enabled=autocast_enabled):
                pred = self.dit(
                    [latent for latent in latents],
                    timesteps.to(device=self.device),
                    context,
                    seq_len,
                    self_attn_mask=self_attn_mask,
                )
        finally:
            for handle in handles:
                handle.remove()

        pred = torch.stack(pred, dim=0).to(device=latents.device, dtype=latents.dtype)
        if not return_hidden_states:
            return pred, None
        return pred, tuple(hidden_states)

    @staticmethod
    def shifted_sigma(u, shift):
        shift = float(shift)
        return shift * u / (1.0 + (shift - 1.0) * u)

    def inference_schedule(self, num_steps, shift=None, device=None, dtype=None):
        if num_steps <= 0:
            raise ValueError(f"`num_steps` must be positive, got {num_steps}")
        device = self.device if device is None else device
        dtype = self.torch_dtype if dtype is None else dtype
        shift = 5.0 if shift is None else float(shift)
        u_steps = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device, dtype=torch.float32)
        sigmas = self.shifted_sigma(u_steps, shift)
        timesteps = sigmas[:-1] * 1000.0
        deltas = sigmas[1:] - sigmas[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def latent_shape(self, batch_size, num_frames, height, width, device=None, dtype=None):
        if num_frames % self.temporal_downsample_factor != 1:
            raise ValueError(f"WAN video frames must satisfy T % 4 == 1, got {num_frames}.")
        if height % self.spatial_downsample_factor != 0 or width % self.spatial_downsample_factor != 0:
            raise ValueError(
                "WAN video spatial dims must be multiples of 16, "
                f"got H={height}, W={width}."
            )
        latent_t = (int(num_frames) - 1) // self.temporal_downsample_factor + 1
        latent_h = int(height) // self.spatial_downsample_factor
        latent_w = int(width) // self.spatial_downsample_factor
        return (
            int(batch_size),
            self.latent_channels,
            latent_t,
            latent_h,
            latent_w,
        )
