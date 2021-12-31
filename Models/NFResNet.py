import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


def formatName(prefix, name):
    return (
        ("%s_%s" % (prefix, name)) if prefix is not None and name is not None else None
    )


class SReLU(tf.keras.layers.ReLU):
    def build(self, input_shape):
        super(SReLU, self).build(input_shape)
        self.gamma = 1.7139588594436646
        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def call(self, x):
        x = super(SReLU, self).call(x)
        return x * self.gamma


class StochDepth(tf.keras.Model):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super(StochDepth, self).__init__(name=name)
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
    def build(self, input_shape):
        super(SkipInit, self).build(input_shape)

        self.skip = self.add_weight(
            name="skip",
            shape=(),
            initializer="zeros",
            dtype="float32",
            trainable=True,
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def call(self, x):
        return x * self.skip


class ScaledWSConv2d(tf.keras.layers.Conv2D):
    """Implements the abs/2101.08692 technique.
    You can simply replace any Conv2D with this one to use re-parametrized
    convolution operation in which the kernels are standardized before conv.
    """

    def build(self, input_shape):
        super(ScaledWSConv2d, self).build(input_shape)

        self.fan_in = self.kernel.shape[0] * self.kernel.shape[1] * self.kernel.shape[2]
        self.gain = self.add_weight(
            name="gain",
            shape=(self.filters,),
            initializer="ones",
            dtype="float32",
            trainable=True,
        )

        self.built = True

    def convolution_op(self, inputs, kernel):
        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)

        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = tf.math.rsqrt(tf.math.maximum(var * self.fan_in, 1e-4)) * self.gain
        shift = mean * scale
        return super().convolution_op(inputs, kernel * scale - shift)


def HeConv2D(
    x, filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=True, name=None
):
    if kernel_size >= 3:
        padding_name = ("%s_padding" % name) if name is not None else name
        x = layers.ZeroPadding2D(
            padding=(int(kernel_size / 2), int(kernel_size / 2)), name=padding_name
        )(x)
    w_init = tf.keras.initializers.VarianceScaling(1.0, "fan_in", "normal")
    x = ScaledWSConv2d(
        filters,
        kernel_size,
        strides,
        padding="valid",
        use_bias=use_bias,
        kernel_initializer=w_init,
        kernel_regularizer=regularizers.l2(0.00005),
        name=name,
    )(x)
    return x


def NFBlock(
    x, filters=64, alpha=1.0, beta=1.0, strides=1, stochdepth_rate=0.0, prefix=None
):
    in_channels = x.shape[-1]

    out = SReLU(name=formatName(prefix, "relu_01"))(x) * beta

    if strides > 1 or in_channels != filters * 4:
        if strides > 1:
            shortcut = layers.AveragePooling2D(
                padding="same", name=formatName(prefix, "averagepooling2d_shortcut")
            )(out)
        else:
            shortcut = out
        shortcut = HeConv2D(
            shortcut,
            filters * 4,
            kernel_size=1,
            name=formatName(prefix, "conv2d_shortcut"),
        )
    else:
        shortcut = x

    out = HeConv2D(out, filters, kernel_size=1, name=formatName(prefix, "conv2d_01"))

    out = SReLU(name=formatName(prefix, "relu_02"))(out)
    out = HeConv2D(
        out,
        filters,
        kernel_size=3,
        strides=strides,
        name=formatName(prefix, "conv2d_02"),
    )

    out = SReLU(name=formatName(prefix, "relu_03"))(out)
    out = HeConv2D(
        out, filters * 4, kernel_size=1, name=formatName(prefix, "conv2d_03")
    )

    if stochdepth_rate > 0.0:
        out = StochDepth(drop_rate=stochdepth_rate)(out)

    out = SkipInit()(out)

    return out * alpha + shortcut


def NFResNet50V1(in_shape=(320, 320, 3), out_classes=2000):
    alpha = 0.2
    stochdepth_rate = 0.1
    definition = {"blocks": [3, 4, 6, 3], "filters": [64, 128, 256, 512]}

    num_blocks = sum(definition["blocks"])

    img_input = layers.Input(shape=in_shape)

    # Root block / "stem"
    x = HeConv2D(
        img_input,
        filters=64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        name="root_conv2d_01",
    )
    x = layers.ZeroPadding2D(padding=(1, 1), name="root_maxpooling2d_01_pad")(x)
    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="valid", name="root_maxpooling2d_01"
    )(x)

    index = 0
    full_index = 0
    expected_std = 1.0
    for stage_depth, block_width in zip(definition["blocks"], definition["filters"]):
        for block_index in range(stage_depth):
            beta = 1.0 / expected_std
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            x = NFBlock(
                x,
                block_width,
                alpha,
                beta,
                strides=2 if (block_index == 0 and index > 0) else 1,
                stochdepth_rate=block_stochdepth_rate,
                prefix="block%d_cell%d" % (index, block_index),
            )

            if block_index == 0:
                expected_std = 1.0
            expected_std = np.sqrt(expected_std ** 2 + alpha ** 2)
            full_index += 1
        index += 1

    # Classification block
    x = SReLU(name="predictions_relu")(x)
    x = layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(out_classes, kernel_initializer="zeros", name="predictions_dense")(
        x
    )
    x = layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="NFResNet50V1")
    return model
