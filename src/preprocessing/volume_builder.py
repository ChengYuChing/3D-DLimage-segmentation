"""Convert ordered 2D frames into overlapping 3D model inputs."""

from __future__ import annotations

import numpy as np


def build_3d_windows(
    sequence: np.ndarray,
    depth: int = 8,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping 3D windows and preserve their original slice mapping."""
    sequence = np.asarray(sequence)

    if sequence.ndim != 3:
        raise ValueError(
            f"Expected [num_slices, height, width], got {sequence.shape}."
        )
    if depth <= 0 or stride <= 0:
        raise ValueError("depth and stride must be positive integers.")
    if sequence.shape[0] < depth:
        raise ValueError("Sequence is shorter than the requested window depth.")

    starts = np.arange(
        0,
        sequence.shape[0] - depth + 1,
        stride,
        dtype=np.int32,
    )
    windows = [sequence[start : start + depth] for start in starts]

    volumes = np.stack(windows, axis=0)            # [B, D, H, W]
    volumes = np.transpose(volumes, (0, 2, 3, 1)) # [B, H, W, D]
    volumes = volumes[..., None].astype(np.float32)

    return volumes, starts
