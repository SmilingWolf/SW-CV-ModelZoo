import numpy as np
import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, factor=0.5):
        super().__init__()
        self.in_ch = in_ch
        self.factor = factor

        squeeze_units = int(in_ch * factor)
        self.dense_squeeze = tf.keras.layers.Dense(
            units=squeeze_units, use_bias=True, activation="relu"
        )
        self.dense_excite = tf.keras.layers.Dense(
            units=self.in_ch, use_bias=True, activation="sigmoid"
        )

    def call(self, x):
        squeeze = tf.reduce_mean(x, [1, 2], keepdims=False)

        attn = self.dense_squeeze(squeeze)
        attn = self.dense_excite(attn)

        attn = tf.reshape(attn, (-1, 1, 1, self.in_ch))
        return tf.math.multiply(x, attn)

    def get_config(self):
        config = super().get_config()
        config.update({"in_ch": self.in_ch})
        config.update({"factor": self.factor})
        return config


class ECABlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, gamma=2, b=1):
        super().__init__()
        self.in_ch = in_ch
        self.gamma = gamma
        self.b = b

        t = int(np.abs((np.log2(self.in_ch) + self.b) / self.gamma))
        k_size = t if t % 2 else t + 1

        self.attn_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, k_size),
            padding="same",
            use_bias=False,
            kernel_initializer="glorot_normal",
            activation="sigmoid",
        )

    def call(self, x):
        squeeze = tf.reduce_mean(x, [1, 2], keepdims=True)
        squeeze = tf.transpose(squeeze, [0, 1, 3, 2])

        attn = self.attn_conv(squeeze)

        attn = tf.transpose(attn, [0, 1, 3, 2])
        return tf.math.multiply(x, attn)

    def get_config(self):
        config = super().get_config()
        config.update({"in_ch": self.in_ch})
        config.update({"gamma": self.gamma})
        config.update({"b": self.b})
        return config
