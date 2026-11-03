"""Generic Cartesian-to-polar transform for the public portfolio pipeline.

This sample is intentionally data-agnostic and contains no clinical paths,
identifiers, DICOM metadata, or annotation parsing logic.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def cartesian_to_polar(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    radial_samples: int = 256,
    angular_samples: int = 256,
) -> np.ndarray:
    """Sample a 2D Cartesian image on a radial polar grid."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    height, width = image.shape
    if center is None:
        center = ((height - 1) / 2.0, (width - 1) / 2.0)

    row_center, col_center = center
    max_radius = min(
        radial_samples,
        row_center,
        col_center,
        height - 1 - row_center,
        width - 1 - col_center,
    )

    radius = np.linspace(0.0, max_radius, radial_samples, dtype=np.float32)
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        angular_samples,
        endpoint=False,
        dtype=np.float32,
    )

    rows = row_center + radius[:, None] * np.cos(theta)[None, :]
    cols = col_center + radius[:, None] * np.sin(theta)[None, :]

    polar = map_coordinates(
        image,
        [rows.ravel(), cols.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return polar.reshape(radial_samples, angular_samples)
