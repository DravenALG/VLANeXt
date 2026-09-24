import torch
import torch.nn as nn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def build_1d_position_ids(batch_size, seq_len, device, start_positions=None, dtype=torch.long):
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    if start_positions is not None:
        positions = positions + start_positions.to(device=device, dtype=dtype).view(batch_size, 1)
    return positions.unsqueeze(0).expand(3, -1, -1)


def build_action_position_ids_after_vlm(vlm_position_ids, action_len, device):
    if vlm_position_ids is None:
        raise ValueError("vlm_position_ids are required when policy_pos_embed='rope'.")
    if vlm_position_ids.ndim != 3 or vlm_position_ids.shape[0] != 3:
        raise ValueError(
            "vlm_position_ids must have shape [3, batch, seq_len], "
            f"got {tuple(vlm_position_ids.shape)}."
        )
    start_positions = vlm_position_ids.to(device=device).amax(dim=(0, 2)) + 1
    return build_1d_position_ids(
        batch_size=vlm_position_ids.shape[1],
        seq_len=action_len,
        device=device,
        start_positions=start_positions,
        dtype=vlm_position_ids.dtype,
    )


def _scaled_mrope_section(mrope_section, half_dim):
    if half_dim < 3:
        raise ValueError(f"3D RoPE requires head_dim // 2 >= 3, got {half_dim}.")
    if mrope_section is None:
        base = [1, 1, 1]
    else:
        base = [int(x) for x in mrope_section]
        if len(base) != 3:
            raise ValueError(f"mrope_section must have 3 values, got {mrope_section}.")

    total = sum(base)
    if total <= 0:
        raise ValueError(f"mrope_section values must be positive, got {mrope_section}.")
    if total == half_dim:
        return base

    raw = [half_dim * x / total for x in base]
    scaled = [max(1, int(x)) for x in raw]
    while sum(scaled) > half_dim:
        idx = max(range(3), key=lambda i: scaled[i])
        scaled[idx] -= 1
    remainder = half_dim - sum(scaled)
    order = sorted(range(3), key=lambda i: raw[i] - scaled[i], reverse=True)
    for i in order[:remainder]:
        scaled[i] += 1
    return scaled


class Qwen3DActionRoPE(nn.Module):
    def __init__(self, head_dim, rope_theta=10000.0, mrope_section=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

        self.head_dim = int(head_dim)
        self.half_dim = self.head_dim // 2
        self.mrope_section = _scaled_mrope_section(mrope_section, self.half_dim)

        inv_freq = 1.0 / (
            float(rope_theta)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_interleaved_mrope(self, freqs):
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                "position_ids must have shape [3, batch, seq_len] or [batch, seq_len], "
                f"got {tuple(position_ids.shape)}."
            )

        inv_freq = self.inv_freq.to(device=x.device)
        position_ids = position_ids.to(device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    def apply_rope(self, x, position_ids):
        cos, sin = self.forward(x, position_ids)
        return apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1)
