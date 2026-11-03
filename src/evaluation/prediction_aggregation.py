"""Aggregate overlapping 3D-window predictions back to an ordered sequence."""

from __future__ import annotations

import numpy as np


def aggregate_overlapping_predictions(
    window_predictions: np.ndarray,
    start_indices: np.ndarray,
    num_slices: int,
) -> np.ndarray:
    """Average overlapping probabilities at their original slice positions."""
    predictions = np.asarray(
        window_predictions,
        dtype=np.float32,
    )

    if (
        predictions.ndim == 5
        and predictions.shape[-1] == 1
    ):
        predictions = predictions[..., 0]

    if predictions.ndim != 4:
        raise ValueError(
            f"Expected [B,H,W,D] predictions, "
            f"got {predictions.shape}."
        )

    starts = np.asarray(
        start_indices,
        dtype=np.int32,
    )

    if len(starts) != predictions.shape[0]:
        raise ValueError(
            "start_indices must match "
            "the number of windows."
        )

    _, height, width, depth = predictions.shape

    probability_sum = np.zeros(
        (height, width, num_slices),
        dtype=np.float32,
    )
    contribution_count = np.zeros(
        (num_slices,),
        dtype=np.float32,
    )

    for window, start in zip(
        predictions,
        starts,
    ):
        end = min(
            start + depth,
            num_slices,
        )
        valid_depth = end - start

        probability_sum[:, :, start:end] += (
            window[:, :, :valid_depth]
        )
        contribution_count[start:end] += 1.0

    if np.any(contribution_count == 0):
        raise ValueError(
            "At least one original slice "
            "received no prediction."
        )

    return (
        probability_sum
        / contribution_count[None, None, :]
    )
