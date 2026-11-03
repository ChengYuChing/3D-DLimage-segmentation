"""Simplified Graph Structural Filtering (GSF) portfolio sample.

The public implementation demonstrates the core design pattern:
normalized graph Laplacian -> graph Fourier transform -> learnable
frequency reweighting -> inverse transform -> residual feature fusion.

It intentionally omits unpublished training code, experiment wiring,
and project-specific data dependencies.
"""

from __future__ import annotations

import tensorflow as tf


class FrequencySE(tf.keras.layers.Layer):
    """Squeeze-and-excitation over graph-frequency components."""

    def __init__(
        self,
        num_frequencies: int,
        reduction: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_frequencies = int(num_frequencies)
        self.reduction = int(reduction)

        hidden = max(
            self.num_frequencies // self.reduction,
            1,
        )
        self.fc1 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.fc2 = tf.keras.layers.Dense(
            self.num_frequencies,
            activation="sigmoid",
        )

    def call(self, spectral_features: tf.Tensor) -> tf.Tensor:
        # spectral_features: [B, F, C]
        descriptor = tf.reduce_mean(
            spectral_features,
            axis=-1,
        )
        weights = self.fc2(self.fc1(descriptor))
        return spectral_features * weights[..., None]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_frequencies": self.num_frequencies,
                "reduction": self.reduction,
            }
        )
        return config


class GraphStructuralFiltering(tf.keras.layers.Layer):
    """Refine node features in the graph-frequency domain.

    The provided graph topology is kept fixed. GSF uses the adjacency
    to define graph frequencies, reweights graph-frequency feature
    components, reconstructs node features, and applies residual fusion.
    """

    def __init__(
        self,
        num_nodes: int,
        reduction: int = 16,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_nodes = int(num_nodes)
        self.reduction = int(reduction)
        self.epsilon = float(epsilon)
        self.frequency_se = FrequencySE(
            self.num_nodes,
            reduction=self.reduction,
        )

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
        node_features, adjacency = inputs

        adjacency = tf.cast(adjacency, tf.float32)
        adjacency = 0.5 * (
            adjacency
            + tf.transpose(adjacency, [0, 2, 1])
        )

        degree = tf.reduce_sum(adjacency, axis=-1)
        d_inv_sqrt = tf.linalg.diag(
            tf.math.rsqrt(
                tf.maximum(degree, 1e-10)
            )
        )
        identity = tf.eye(
            tf.shape(adjacency)[1],
            batch_shape=[tf.shape(adjacency)[0]],
        )

        laplacian = (
            identity
            - tf.matmul(
                tf.matmul(d_inv_sqrt, adjacency),
                d_inv_sqrt,
            )
        )
        laplacian = 0.5 * (
            laplacian
            + tf.transpose(laplacian, [0, 2, 1])
        )
        laplacian = laplacian + self.epsilon * identity

        _, eigenvectors = tf.linalg.eigh(laplacian)

        # Graph Fourier transform: U^T X
        spectral = tf.einsum(
            "bij,bjk->bik",
            tf.transpose(eigenvectors, [0, 2, 1]),
            node_features,
        )
        spectral = self.frequency_se(spectral)

        # Inverse graph Fourier transform: U X_hat
        reconstructed = tf.einsum(
            "bij,bjk->bik",
            eigenvectors,
            spectral,
        )

        return node_features + reconstructed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_nodes": self.num_nodes,
                "reduction": self.reduction,
                "epsilon": self.epsilon,
            }
        )
        return config
