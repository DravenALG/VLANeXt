from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_LAM_TYPES = {"vae", "vq", "vicreg"}


def _num_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PairEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels * 2, hidden_size, stride=1),
            ConvBlock(hidden_size, hidden_size * 2, stride=2),
            ConvBlock(hidden_size * 2, hidden_size * 4, stride=2),
            ConvBlock(hidden_size * 4, hidden_size * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size * 4, latent_dim),
        )

    def forward(self, frame_t, frame_tp1):
        x = torch.cat([frame_t, frame_tp1], dim=1)
        return self.proj(self.net(x))


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels, hidden_size, latent_dim):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, hidden_size, stride=1)
        self.enc2 = ConvBlock(hidden_size, hidden_size * 2, stride=2)
        self.enc3 = ConvBlock(hidden_size * 2, hidden_size * 4, stride=2)
        self.action_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size * 4, hidden_size * 4),
        )
        self.up2 = nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(hidden_size * 4, hidden_size * 2, stride=1)
        self.up1 = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(hidden_size * 2, hidden_size, stride=1)
        self.out = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_size, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, frame_t, action):
        h1 = self.enc1(frame_t)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        cond = self.action_proj(action).to(dtype=h3.dtype).unsqueeze(-1).unsqueeze(-1)
        h3 = h3 + cond
        y = self.up2(h3)
        y = self._match_hw(y, h2)
        y = self.dec2(torch.cat([y, h2], dim=1))
        y = self.up1(y)
        y = self._match_hw(y, h1)
        y = self.dec1(torch.cat([y, h1], dim=1))
        return self.out(y)

    @staticmethod
    def _match_hw(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, latent_dim, commitment_weight=0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.commitment_weight = float(commitment_weight)
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        bound = 1.0 / max(1, self.codebook_size)
        nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, z_e):
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss
        z_st = z_e + (z_q - z_e).detach()
        return z_st, loss, indices

    @torch.no_grad()
    def quantize(self, z_e):
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        return self.embedding(indices), indices


class LatentActionModel(nn.Module):
    def __init__(
        self,
        *,
        lam_type="vae",
        input_channels=3,
        latent_dim=7,
        hidden_size=96,
        kl_weight=1.0e-4,
        codebook_size=128,
        commitment_weight=0.25,
        vq_weight=1.0,
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        recon_l1_weight=1.0,
        recon_mse_weight=0.1,
        min_std=1.0,
        eps=1.0e-4,
    ):
        super().__init__()
        lam_type = str(lam_type).lower()
        if lam_type not in SUPPORTED_LAM_TYPES:
            raise ValueError(f"Unknown lam_type={lam_type}. Options: {sorted(SUPPORTED_LAM_TYPES)}")
        self.lam_type = lam_type
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.kl_weight = float(kl_weight)
        self.vq_weight = float(vq_weight)
        self.vicreg_invariance_weight = float(vicreg_invariance_weight)
        self.vicreg_variance_weight = float(vicreg_variance_weight)
        self.vicreg_covariance_weight = float(vicreg_covariance_weight)
        self.recon_l1_weight = float(recon_l1_weight)
        self.recon_mse_weight = float(recon_mse_weight)
        self.min_std = float(min_std)
        self.eps = float(eps)

        enc_out_dim = self.latent_dim * 2 if self.lam_type == "vae" else self.latent_dim
        self.encoder = PairEncoder(self.input_channels, self.hidden_size, enc_out_dim)
        self.decoder = ConditionalDecoder(self.input_channels, self.hidden_size, self.latent_dim)
        self.quantizer = None
        if self.lam_type == "vq":
            self.quantizer = VectorQuantizer(codebook_size, self.latent_dim, commitment_weight)

    def forward(self, frame_t, frame_tp1, frame_t_aug=None, frame_tp1_aug=None):
        encoded = self.encoder(frame_t, frame_tp1)
        extras = {}
        if self.lam_type == "vae":
            mu, logvar = encoded.chunk(2, dim=1)
            action = self._reparameterize(mu, logvar)
            extras["mu"] = mu
            extras["logvar"] = logvar
            constraint_loss = self._kl_loss(mu, logvar) * self.kl_weight
            constraint_name = "kl_loss"
        elif self.lam_type == "vq":
            action, vq_loss, indices = self.quantizer(encoded)
            extras["code_indices"] = indices
            constraint_loss = vq_loss * self.vq_weight
            constraint_name = "vq_loss"
        else:
            action = encoded
            constraint_loss = self._vicreg_loss(action, frame_t_aug, frame_tp1_aug)
            constraint_name = "vicreg_loss"

        recon = self.decoder(frame_t, action)
        recon_loss = self._reconstruction_loss(recon, frame_tp1)
        loss = recon_loss + constraint_loss
        out = {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            constraint_name: constraint_loss.detach(),
            "latent_action": action.detach(),
            "reconstruction": recon,
        }
        out.update(extras)
        return out

    @torch.no_grad()
    def encode_action(self, frame_t, frame_tp1, deterministic=True, return_indices=False):
        encoded = self.encoder(frame_t, frame_tp1)
        indices = None
        if self.lam_type == "vae":
            mu, logvar = encoded.chunk(2, dim=1)
            action = mu if deterministic else self._reparameterize(mu, logvar)
        elif self.lam_type == "vq":
            action, indices = self.quantizer.quantize(encoded)
        else:
            action = encoded
        if return_indices:
            return action, indices
        return action

    def decode_next(self, frame_t, latent_action):
        return self.decoder(frame_t, latent_action)

    def _reconstruction_loss(self, recon, target):
        loss = recon.new_tensor(0.0)
        if self.recon_l1_weight > 0:
            loss = loss + self.recon_l1_weight * F.l1_loss(recon, target)
        if self.recon_mse_weight > 0:
            loss = loss + self.recon_mse_weight * F.mse_loss(recon, target)
        return loss

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_loss(mu, logvar):
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    def _vicreg_loss(self, z, frame_t_aug, frame_tp1_aug):
        if frame_t_aug is None or frame_tp1_aug is None:
            z_aug = z
        else:
            z_aug = self.encoder(frame_t_aug, frame_tp1_aug)
        inv = F.mse_loss(z, z_aug)
        var = self._variance_loss(z) + self._variance_loss(z_aug)
        cov = self._covariance_loss(z) + self._covariance_loss(z_aug)
        return (
            self.vicreg_invariance_weight * inv
            + self.vicreg_variance_weight * var
            + self.vicreg_covariance_weight * cov
        )

    def _variance_loss(self, z):
        if z.shape[0] <= 1:
            return z.new_tensor(0.0)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(self.min_std - std))

    @staticmethod
    def _covariance_loss(z):
        if z.shape[0] <= 1:
            return z.new_tensor(0.0)
        z = z - z.mean(dim=0)
        cov = (z.t() @ z) / max(1, z.shape[0] - 1)
        off_diag = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[0] + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / z.shape[1]


def _model_config(config):
    return config.get("model", config)


def build_lam_from_config(config):
    cfg = _model_config(config)
    view_mode = config.get("data", {}).get("view_mode", "single") if isinstance(config, dict) else "single"
    default_channels = 6 if view_mode == "multi" else 3
    return LatentActionModel(
        lam_type=cfg.get("lam_type", cfg.get("type", "vae")),
        input_channels=cfg.get("input_channels", default_channels),
        latent_dim=cfg["latent_dim"],
        hidden_size=cfg.get("hidden_size", 96),
        kl_weight=cfg.get("kl_weight", 1.0e-4),
        codebook_size=cfg.get("codebook_size", 128),
        commitment_weight=cfg.get("commitment_weight", 0.25),
        vq_weight=cfg.get("vq_weight", 1.0),
        vicreg_invariance_weight=cfg.get("vicreg_invariance_weight", 25.0),
        vicreg_variance_weight=cfg.get("vicreg_variance_weight", 25.0),
        vicreg_covariance_weight=cfg.get("vicreg_covariance_weight", 1.0),
        recon_l1_weight=cfg.get("recon_l1_weight", 1.0),
        recon_mse_weight=cfg.get("recon_mse_weight", 0.1),
        min_std=cfg.get("min_std", 1.0),
    )


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if not first_key.startswith("module."):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_lam_checkpoint(checkpoint_path, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load(str(checkpoint_path), map_location=map_location)
    config = checkpoint.get("config")
    if config is None:
        config = {"model": checkpoint.get("model_config", {})}
    model = build_lam_from_config(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    return model, config, checkpoint
