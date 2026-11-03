"""Residual graph message-passing layer used by the model family."""

from __future__ import annotations

import tensorflow as tf


class ResidualGraphConv(tf.keras.layers.Layer):
    """Aggregate neighbor messages and preserve a residual node representation."""

    def __init__(
        self,
        hidden_units: int = 256,
        dropout_rate: float = 0.2,
        normalize_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = int(hidden_units)
        self.dropout_rate = float(dropout_rate)
        self.normalize_output = bool(normalize_output)

        self.node_projection = tf.keras.layers.Dense(
            self.hidden_units,
            activation=None,
        )
        self.message_projection = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(
                    self.hidden_units,
                    activation="leaky_relu",
                ),
            ]
        )
        self.update_projection = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(
                    self.hidden_units,
                    activation="leaky_relu",
                ),
            ]
        )

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
        node_features, adjacency = inputs

        residual = self.node_projection(node_features)
        messages = self.message_projection(node_features)
        aggregated = tf.matmul(adjacency, messages)
        updated = self.update_projection(residual + aggregated)

        if self.normalize_output:
            updated = tf.nn.l2_normalize(updated, axis=-1)

        return updated

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "normalize_output": self.normalize_output,
            }
        )
        return config
