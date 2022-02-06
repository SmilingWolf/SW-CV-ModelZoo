import tensorflow as tf


def format_name(prefix, name):
    return (
        ("%s_%s" % (prefix, name)) if prefix is not None and name is not None else None
    )


def inception_scaling(x):
    return (tf.cast(x, tf.float32) - 127.5) * (1 / 127.5)


def simple_scaling(x):
    return tf.cast(x, tf.float32) * (1 / 255.0)


def input_scaling(method="inception"):
    if method == "inception":
        return inception_scaling
    elif method == "simple":
        return simple_scaling
    elif method == None:
        return lambda x: tf.cast(x, tf.float32)


scaled_acts = {
    "relu": lambda x: tf.nn.relu(x) * 1.7139588594436646,
    "relu6": lambda x: tf.nn.relu6(x) * 1.7131484746932983,
    "silu": lambda x: tf.nn.silu(x) * 1.7881293296813965,
    "hswish": lambda x: (x * tf.nn.relu6(x + 3) * 0.16666666666666667)
    * 1.8138962328745718,
    "sigmoid": lambda x: tf.nn.sigmoid(x) * 4.803835391998291,
}


class StochDepth(tf.keras.Model):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, x, training):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        r = tf.random.uniform(shape=[batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class SkipInit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skip = self.add_weight(
            name="skip",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x * self.skip
