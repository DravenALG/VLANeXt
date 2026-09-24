"""Utilities for converting action targets into language-action text."""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - scipy is expected in training envs.
    Rotation = None


LANGUAGE_ACTION_ALIASES = {
    "vla0": "vla-0",
    "vla_0": "vla-0",
    "vla-0": "vla-0",
    "vla0_chunked": "vla-0",
    "lap": "lap",
    "language-action": "lap",
    "language_action": "lap",
}


# normalize the prompt for VLM input for language-action stype learning
def format_language_action_prompt(
    instruction: str,
    *,
    language_action_format: str,
    frame_description: str = "robot base frame",
    future_len: int | None = None,
    action_dim: int | None = None,
) -> str:
    fmt = normalize_language_action_format(language_action_format)
    instruction = str(instruction).strip().replace("_", " ").replace("\n", " ").rstrip(".")
    if fmt == "vla-0":
        horizon = int(future_len or 1)
        dim = int(action_dim or 7)
        return (
            f"Task: {instruction}. Predict robot actions for the next {horizon} timesteps. "
            f"Each action has {dim} dimensions. Output only space-separated integers."
        )
    return f"Task: {instruction}, predict the robot's action in the {frame_description}."


def normalize_language_action_format(name: str | None) -> str:
    key = (name or "lap").strip().lower()
    if key not in LANGUAGE_ACTION_ALIASES:
        raise ValueError(
            f"Unknown language_action_format: {name!r}. "
            "Options: 'vla-0', 'lap'/'language-action'."
        )
    return LANGUAGE_ACTION_ALIASES[key]


# compute movement actions for LAP-style language-action formatting
def build_libero_lap_language_actions(raw_state: np.ndarray, raw_actions: np.ndarray) -> np.ndarray:
    """Build per-timestep LAP-style movement actions in physical units."""
    eef_state = libero_raw_state_to_eef_state(raw_state)
    movement = compute_padded_movement_actions(eef_state[:, :6])

    gripper = np.asarray(raw_actions[:, 6:7], dtype=np.float32)
    # LIBERO action: -1=open, +1=close. LAP text formatter expects 1=open, 0=close.
    gripper_open = 1.0 - np.clip(gripper, 0.0, 1.0)
    return np.concatenate([movement, gripper_open], axis=1).astype(np.float32)


def libero_raw_state_to_eef_state(raw_state: np.ndarray) -> np.ndarray:
    """Convert LIBERO raw state to [xyz, extrinsic-XYZ euler, gripper_open]."""
    raw_state = np.asarray(raw_state, dtype=np.float32)
    euler_xyz = _axis_angle_to_extrinsic_xyz_euler(raw_state[:, 3:6])
    gripper_state = np.clip(raw_state[:, -2:-1] / 0.04, 0.0, 1.0)
    return np.concatenate([raw_state[:, :3], euler_xyz, gripper_state], axis=1)


def _axis_angle_to_extrinsic_xyz_euler(axis_angle: np.ndarray) -> np.ndarray:
    if Rotation is None:
        raise ImportError("scipy is required for LAP language-action rotation conversion.")
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    original_shape = axis_angle.shape[:-1]
    rot = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot = rot.reshape(*original_shape, 3, 3)

    r20 = rot[..., 2, 0]
    r21 = rot[..., 2, 1]
    r22 = rot[..., 2, 2]
    r10 = rot[..., 1, 0]
    r00 = rot[..., 0, 0]
    pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
    roll = np.arctan2(r21, r22)
    yaw = np.arctan2(r10, r00)
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def compute_padded_movement_actions(eef_state: np.ndarray) -> np.ndarray:
    """Compute action[t] = state[t+1] - state[t], with a final zero movement."""
    eef_state = np.asarray(eef_state, dtype=np.float32)
    if eef_state.shape[0] <= 1:
        return np.zeros((eef_state.shape[0], 6), dtype=np.float32)

    delta_xyz = eef_state[1:, :3] - eef_state[:-1, :3]
    delta_rpy = _euler_diff(eef_state[1:, 3:6], eef_state[:-1, 3:6])
    deltas = np.concatenate([delta_xyz, delta_rpy], axis=1)
    return np.concatenate([deltas, np.zeros((1, 6), dtype=np.float32)], axis=0)


def _euler_diff(next_euler: np.ndarray, current_euler: np.ndarray) -> np.ndarray:
    r_next = _euler_to_rotation_matrix(next_euler)
    r_current = _euler_to_rotation_matrix(current_euler)
    r_rel = np.matmul(np.swapaxes(r_current, -1, -2), r_next)
    return _rotation_matrix_to_euler(r_rel)


# generate language-action text in the style of the VLA-0 paper
def format_vla0_action_text(actions, num_bins: int = 1000) -> str:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    actions = np.clip(actions, -1.0, 1.0)
    values = np.rint((actions + 1.0) * 0.5 * int(num_bins)).astype(np.int64)
    values = np.clip(values, 0, int(num_bins)).reshape(-1)
    return " ".join(str(int(v)) for v in values)


# generate language-action text in the style of the LAP paper
def summarize_lap_language_action(actions, decimal_places: int = 0) -> str:
    """Summarize one action window as LAP-style language-action text."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[-1] < 7:
        raise ValueError(f"LAP language-action requires 7D actions, got shape {actions.shape}")

    dx_m = float(actions[:, 0].sum())
    dy_m = float(actions[:, 1].sum())
    dz_m = float(actions[:, 2].sum())
    dx = round(abs(dx_m * 100.0), decimal_places)
    dy = round(abs(dy_m * 100.0), decimal_places)
    dz = round(abs(dz_m * 100.0), decimal_places)

    rotation_total = np.eye(3, dtype=np.float64)
    for rpy in actions[:, 3:6]:
        rotation_total = rotation_total @ _euler_to_rotation_matrix(rpy)
    droll_rad, dpitch_rad, dyaw_rad = _rotation_matrix_to_euler(rotation_total)
    droll_rad = float(droll_rad)
    dpitch_rad = float(dpitch_rad)
    dyaw_rad = float(dyaw_rad)
    droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), 10)
    dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), 10)
    dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), 10)

    parts: list[str] = []
    fmt_dx = _format_numeric(dx, decimal_places)
    fmt_dy = _format_numeric(dy, decimal_places)
    fmt_dz = _format_numeric(dz, decimal_places)
    if dx_m > 0 and dx != 0:
        parts.append(f"move forward {fmt_dx} cm")
    elif dx_m < 0 and dx != 0:
        parts.append(f"move back {fmt_dx} cm")
    if dz_m > 0 and dz != 0:
        parts.append(f"move up {fmt_dz} cm")
    elif dz_m < 0 and dz != 0:
        parts.append(f"move down {fmt_dz} cm")
    if dy_m > 0 and dy != 0:
        parts.append(f"move left {fmt_dy} cm")
    elif dy_m < 0 and dy != 0:
        parts.append(f"move right {fmt_dy} cm")

    if droll_rad > 0 and droll != 0:
        parts.append(f"tilt left {droll} degrees")
    elif droll_rad < 0 and droll != 0:
        parts.append(f"tilt right {droll} degrees")
    if dpitch_rad > 0 and dpitch != 0:
        parts.append(f"tilt back {dpitch} degrees")
    elif dpitch_rad < 0 and dpitch != 0:
        parts.append(f"tilt forward {dpitch} degrees")
    if dyaw_rad > 0 and dyaw != 0:
        parts.append(f"rotate counterclockwise {dyaw} degrees")
    elif dyaw_rad < 0 and dyaw != 0:
        parts.append(f"rotate clockwise {dyaw} degrees")

    if float(actions[-1, 6]) >= 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    return ", ".join(parts)


def _euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(euler, -1, 0)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


def _rotation_matrix_to_euler(rot: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    r00 = rot[..., 0, 0]
    r10 = rot[..., 1, 0]
    r11 = rot[..., 1, 1]
    r12 = rot[..., 1, 2]
    r20 = rot[..., 2, 0]
    r21 = rot[..., 2, 1]
    r22 = rot[..., 2, 2]

    sy = np.sqrt(np.maximum(r00 * r00 + r10 * r10, eps))
    singular = sy < eps
    roll = np.where(singular, np.arctan2(-r12, r11), np.arctan2(r21, r22))
    pitch = np.arctan2(-r20, sy)
    yaw = np.where(singular, np.zeros_like(r00), np.arctan2(r10, r00))
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def _round_to_nearest_n(value: float, n: int = 10) -> int:
    return int(round(value / n) * n)


def _format_numeric(value: float, decimal_places: int = 0) -> str:
    return f"{value:.{int(decimal_places)}f}"


# main entry point for formatting language-action text from action targets
def format_language_action_text(
    actions,
    *,
    language_action_format: str,
    num_bins: int = 1000,
) -> str:
    fmt = normalize_language_action_format(language_action_format)
    if fmt == "vla-0":
        return format_vla0_action_text(actions, num_bins=num_bins)
    return summarize_lap_language_action(actions)
