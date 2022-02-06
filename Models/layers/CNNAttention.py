import numpy as np
import tensorflow as tf


def se_block(x, in_ch, factor=0.5):
    squeeze = tf.reduce_mean(x, [1, 2], keepdims=False)

    attn = tf.keras.layers.Dense(units=int(in_ch * factor), use_bias=True)(squeeze)
    attn = tf.keras.layers.ReLU()(attn)
    attn = tf.keras.layers.Dense(units=in_ch, use_bias=True)(attn)

    attn = tf.reshape(attn, (-1, 1, 1, in_ch))
    attn = tf.math.sigmoid(attn)
    return attn


def eca_block(x, in_ch, gamma=2, b=1):
    t = int(np.abs((np.log2(in_ch) + b) / gamma))
    k_size = t if t % 2 else t + 1

    squeeze = tf.reduce_mean(x, [1, 2], keepdims=True)
    squeeze = tf.squeeze(squeeze, axis=2)
    squeeze = tf.transpose(squeeze, [0, 2, 1])

    attn = tf.keras.layers.Conv1D(
        filters=1,
        kernel_size=k_size,
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_normal",
    )(squeeze)

    attn = tf.transpose(attn, [0, 2, 1])
    attn = tf.expand_dims(attn, axis=2)
    attn = tf.math.sigmoid(attn)
    return attn
