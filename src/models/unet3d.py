"""Compact 3D Attention U-Net backbone for the public portfolio sample.

The original research model contained additional experiment-specific
wiring. This version keeps the shared engineering ideas used across
comparisons: 3D convolutions, GroupNorm, anisotropic depth handling,
ASPP-style context, and attention-gated skip connections.
"""

from __future__ import annotations

import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        groups: int = 8,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        groups = min(self.groups, channels)

        if channels % groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible "
                f"by groups={groups}"
            )

        self._groups = groups
        self.gamma = self.add_weight(
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        shape = tf.shape(x)
        b, h, w, d, c = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
        )

        x = tf.reshape(
            x,
            [b, h, w, d, self._groups, c // self._groups],
        )
        mean, variance = tf.nn.moments(
            x,
            axes=[1, 2, 3, 5],
            keepdims=True,
        )
        x = (x - mean) / tf.sqrt(
            variance + self.epsilon
        )
        x = tf.reshape(x, [b, h, w, d, c])

        return x * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "groups": self.groups,
                "epsilon": self.epsilon,
            }
        )
        return config


def conv_block(
    x,
    filters: int,
    groups: int = 8,
    kernel=(3, 3, 3),
):
    for _ in range(2):
        x = tf.keras.layers.Conv3D(
            filters,
            kernel,
            padding="same",
            activation="relu",
        )(x)
        x = GroupNormalization(groups=groups)(x)
    return x


def attention_gate(skip, gate, filters: int):
    gate = tf.keras.layers.UpSampling3D(
        size=(2, 2, 1)
    )(gate)

    skip_projection = tf.keras.layers.Conv3D(
        filters,
        1,
        padding="same",
    )(skip)
    gate_projection = tf.keras.layers.Conv3D(
        filters,
        1,
        padding="same",
    )(gate)

    score = tf.keras.layers.Activation("relu")(
        skip_projection + gate_projection
    )
    score = tf.keras.layers.Conv3D(
        1,
        1,
        padding="same",
        activation="sigmoid",
    )(score)

    return skip * score


def aspp_3d(x, filters: int):
    branches = [
        tf.keras.layers.Conv3D(
            filters,
            1,
            padding="same",
            activation="relu",
        )(x)
    ]

    for rate in (3, 6, 9):
        branches.append(
            tf.keras.layers.Conv3D(
                filters,
                3,
                padding="same",
                dilation_rate=(rate, rate, 1),
                activation="relu",
            )(x)
        )

    x = tf.keras.layers.Concatenate()(branches)

    return tf.keras.layers.Conv3D(
        filters,
        1,
        padding="same",
        activation="relu",
    )(x)


def build_unet3d(
    input_shape=(256, 256, 8, 1),
    base_filters: int = 32,
) -> tf.keras.Model:
    """Build a compact shared 3D Attention U-Net backbone."""
    inputs = tf.keras.Input(shape=input_shape)

    e1 = conv_block(
        inputs,
        base_filters,
        groups=8,
        kernel=(3, 3, 1),
    )
    p1 = tf.keras.layers.MaxPool3D((2, 2, 1))(e1)

    e2 = conv_block(
        p1,
        base_filters * 2,
        groups=8,
    )
    p2 = tf.keras.layers.MaxPool3D((2, 2, 1))(e2)

    e3 = conv_block(
        p2,
        base_filters * 4,
        groups=16,
    )
    p3 = tf.keras.layers.MaxPool3D((2, 2, 1))(e3)

    e4 = conv_block(
        p3,
        base_filters * 8,
        groups=16,
    )
    p4 = tf.keras.layers.MaxPool3D((2, 2, 1))(e4)

    bottleneck = aspp_3d(
        p4,
        base_filters * 16,
    )
    bottleneck = tf.keras.layers.Conv3D(
        base_filters * 16,
        3,
        padding="same",
        dilation_rate=(1, 1, 2),
        activation="relu",
    )(bottleneck)

    x = bottleneck

    for skip, filters, groups in [
        (e4, base_filters * 8, 16),
        (e3, base_filters * 4, 16),
        (e2, base_filters * 2, 8),
        (e1, base_filters, 8),
    ]:
        attended = attention_gate(
            skip,
            x,
            filters,
        )

        x = tf.keras.layers.Conv3DTranspose(
            filters,
            (2, 2, 1),
            strides=(2, 2, 1),
            padding="same",
        )(x)

        x = tf.keras.layers.Concatenate()(
            [x, attended]
        )

        kernel = (
            (3, 3, 1)
            if filters == base_filters
            else (3, 3, 3)
        )
        x = conv_block(
            x,
            filters,
            groups=groups,
            kernel=kernel,
        )

    outputs = tf.keras.layers.Conv3D(
        1,
        1,
        activation="sigmoid",
        dtype="float32",
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="portfolio_3d_attention_unet",
    )
