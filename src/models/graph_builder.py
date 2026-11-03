"""Feature-space KNN graph construction for compact 3D representations."""

from __future__ import annotations

import tensorflow as tf


class KNNGraph(tf.keras.layers.Layer):
    """Build a symmetric binary KNN adjacency from node features."""

    def __init__(self, k: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.trainable = False

    def call(self, node_features: tf.Tensor) -> tf.Tensor:
        # node_features: [B, N, C]
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        xt = tf.transpose(node_features, [0, 2, 1])
        squared = tf.reduce_sum(
            tf.square(node_features),
            axis=-1,
            keepdims=True,
        )
        distance = (
            squared
            - 2.0 * tf.matmul(node_features, xt)
            + tf.transpose(squared, [0, 2, 1])
        )

        _, neighbors = tf.nn.top_k(-distance, k=self.k + 1)
        neighbors = neighbors[:, :, 1:]  # remove self-neighbor

        batch_index = tf.repeat(
            tf.range(batch_size),
            num_nodes * self.k,
        )
        row_index = tf.tile(
            tf.repeat(tf.range(num_nodes), self.k),
            [batch_size],
        )
        col_index = tf.reshape(neighbors, [-1])

        indices = tf.stack(
            [
                tf.cast(batch_index, tf.int64),
                tf.cast(row_index, tf.int64),
                tf.cast(col_index, tf.int64),
            ],
            axis=1,
        )

        dense_shape = tf.cast(
            tf.stack([batch_size, num_nodes, num_nodes]),
            tf.int64,
        )
        adjacency = tf.sparse.SparseTensor(
            indices=indices,
            values=tf.ones(tf.shape(indices)[0], dtype=tf.float32),
            dense_shape=dense_shape,
        )
        adjacency = tf.sparse.to_dense(
            tf.sparse.reorder(adjacency)
        )

        return tf.maximum(
            adjacency,
            tf.transpose(adjacency, [0, 2, 1]),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config
