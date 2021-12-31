import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


def formatName(prefix, name):
    return (
        ("%s_%s" % (prefix, name)) if prefix is not None and name is not None else None
    )


def SEBlock(x, in_ch):
    squeeze = tf.reduce_mean(x, [1, 2], keepdims=False)

    attn = tf.keras.layers.Dense(filters=int(in_ch * 0.5), use_bias=True)(squeeze)
    attn = tf.keras.layers.Relu()(attn)
    attn = tf.keras.layers.Dense(filters=in_ch, use_bias=True)(attn)

    attn = tf.reshape(attn, (-1, 1, 1, in_ch))
    attn = tf.math.sigmoid(attn)
    return attn


def ECABlock(x, in_ch, gamma=2, b=1):
    t = int(np.abs((np.log2(in_ch) + b) / gamma))
    k_size = t if t % 2 else t + 1

    squeeze = tf.reduce_mean(x, [1, 2], keepdims=True)
    squeeze = tf.squeeze(squeeze, axis=2)
    squeeze = tf.transpose(squeeze, [0, 2, 1])

    w_init = tf.keras.initializers.VarianceScaling(1.0, "fan_in", "normal")
    attn = tf.keras.layers.Conv1D(
        filters=1,
        kernel_size=k_size,
        padding="same",
        use_bias=False,
        kernel_initializer=w_init,
    )(squeeze)

    attn = tf.transpose(attn, [0, 2, 1])
    attn = tf.expand_dims(attn, axis=2)
    attn = tf.math.sigmoid(attn)
    return attn


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
    x,
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    use_bias=True,
    groups=1,
    name=None,
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
        groups=groups,
        kernel_initializer=w_init,
        kernel_regularizer=regularizers.l2(0.00005),
        name=name,
    )(x)
    return x


def NFBlock(
    x,
    out_filters=64,
    alpha=1.0,
    beta=1.0,
    strides=1,
    group_size=128,
    expansion=0.5,
    use_eca=True,
    stochdepth_rate=0.0,
    prefix=None,
):
    in_channels = x.shape[-1]

    in_filters = int(out_filters * expansion)
    groups = in_filters // group_size
    in_filters = groups * group_size

    out = SReLU(name=formatName(prefix, "relu_01"))(x) * beta

    if strides > 1 or in_channels != out_filters:
        if strides > 1:
            shortcut = layers.AveragePooling2D(
                padding="same", name=formatName(prefix, "averagepooling2d_shortcut")
            )(out)
        else:
            shortcut = out
        shortcut = HeConv2D(
            shortcut,
            out_filters,
            kernel_size=1,
            name=formatName(prefix, "conv2d_shortcut"),
        )
    else:
        shortcut = x

    out = HeConv2D(out, in_filters, kernel_size=1, name=formatName(prefix, "conv2d_01"))

    out = SReLU(name=formatName(prefix, "relu_02"))(out)
    out = HeConv2D(
        out,
        in_filters,
        kernel_size=3,
        strides=strides,
        groups=groups,
        name=formatName(prefix, "conv2d_02"),
    )

    out = SReLU(name=formatName(prefix, "relu_03"))(out)
    out = HeConv2D(
        out,
        in_filters,
        kernel_size=3,
        groups=groups,
        name=formatName(prefix, "conv2d_03"),
    )

    out = SReLU(name=formatName(prefix, "relu_04"))(out)
    out = HeConv2D(
        out, out_filters, kernel_size=1, name=formatName(prefix, "conv2d_04")
    )
    if use_eca:
        out = 2 * ECABlock(out, out_filters) * out  # Multiply by 2 for rescaling

    if stochdepth_rate > 0.0:
        out = StochDepth(drop_rate=stochdepth_rate)(out)

    out = SkipInit()(out)

    return out * alpha + shortcut


# Lx variants params from TIMM: "experimental 'light' versions of NFNet-F that are little leaner"
definitions = {
    "F0": {
        "blocks": [1, 2, 6, 3],
        "filters": [256, 512, 1536, 1536],
        "group_size": 128,
        "drop_rate": 0.2,
        "bneck_expansion": 0.5,
        "final_expansion": 2,
    },
    "L0": {
        "blocks": [1, 2, 6, 3],
        "filters": [256, 512, 1536, 1536],
        "group_size": 64,
        "drop_rate": 0.2,
        "bneck_expansion": 0.25,
        "final_expansion": 1.5,
    },
    "L1": {
        "blocks": [2, 4, 12, 6],
        "filters": [256, 512, 1536, 1536],
        "group_size": 64,
        "drop_rate": 0.3,
        "bneck_expansion": 0.25,
        "final_expansion": 2,
    },
}


def NFNetV1(
    in_shape=(320, 320, 3), out_classes=2000, definition_name="L0", use_eca=True
):
    alpha = 0.2
    width = 1.0
    stochdepth_rate = 0.1

    definition = definitions[definition_name]
    strides = [1, 2, 2, 2]

    num_blocks = sum(definition["blocks"])

    img_input = layers.Input(shape=in_shape)

    # Root block / "stem"
    ch = definition["filters"][0] // 2
    x = HeConv2D(img_input, filters=16, kernel_size=3, strides=2, name="root_conv2d_01")
    x = SReLU(name="root_relu_01")(x)
    x = HeConv2D(x, filters=32, kernel_size=3, strides=1, name="root_conv2d_02")
    x = SReLU(name="root_relu_02")(x)
    x = HeConv2D(x, filters=64, kernel_size=3, strides=1, name="root_conv2d_03")
    x = SReLU(name="root_relu_03")(x)
    x = HeConv2D(x, filters=ch, kernel_size=3, strides=2, name="root_conv2d_04")

    index = 0
    full_index = 0
    expected_std = 1.0
    for stage_depth, block_width, stride in zip(
        definition["blocks"], definition["filters"], strides
    ):
        for block_index in range(stage_depth):
            beta = 1.0 / expected_std
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            out_ch = int(block_width * width)
            x = NFBlock(
                x,
                out_ch,
                alpha,
                beta,
                strides=stride if block_index == 0 else 1,
                group_size=definition["group_size"],
                expansion=definition["bneck_expansion"],
                use_eca=use_eca,
                stochdepth_rate=block_stochdepth_rate,
                prefix="block%d_cell%d" % (index, block_index),
            )

            ch = out_ch
            if block_index == 0:
                expected_std = 1.0
            expected_std = np.sqrt(expected_std ** 2 + alpha ** 2)
            full_index += 1
        index += 1

    # Classification block
    x = HeConv2D(
        x,
        int(ch * definition["final_expansion"]),
        kernel_size=1,
        name="predictions_conv2d",
    )
    x = SReLU(name="predictions_relu")(x)
    x = layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = layers.Dropout(definition["drop_rate"])(x)

    x = layers.Dense(out_classes, kernel_initializer="zeros", name="predictions_dense")(
        x
    )
    x = layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="NFNet%sV1" % definition_name)
    return model
