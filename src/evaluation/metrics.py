"""Binary segmentation metrics used for vessel-level evaluation."""

from __future__ import annotations

import numpy as np


def _binary_masks(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
):
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    pred = prediction >= threshold
    true = target > 0
    return pred, true


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Dice, precision, recall, and IoU for one 3D case."""
    pred, true = _binary_masks(
        prediction,
        target,
        threshold,
    )

    tp = np.count_nonzero(pred & true)
    fp = np.count_nonzero(pred & ~true)
    fn = np.count_nonzero(~pred & true)

    dice_den = 2 * tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    iou_den = tp + fp + fn

    return {
        "dice": (
            np.nan
            if dice_den == 0
            else (2.0 * tp) / dice_den
        ),
        "precision": (
            np.nan
            if precision_den == 0
            else tp / precision_den
        ),
        "recall": (
            np.nan
            if recall_den == 0
            else tp / recall_den
        ),
        "iou": (
            np.nan
            if iou_den == 0
            else tp / iou_den
        ),
    }
