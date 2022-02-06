import tensorflow as tf

from . import Base


class ScaledWSConv2D(tf.keras.layers.Conv2D):
    """
    Implements the abs/2101.08692 technique.
    You can simply replace any Conv2D with this one to use re-parametrized
    convolution operation in which the kernels are standardized before conv.
    """

    def build(self, input_shape):
        super().build(input_shape)

        self.fan_in = self.kernel.shape[0] * self.kernel.shape[1] * self.kernel.shape[2]
        self.gain = self.add_weight(
            name="gain",
            shape=(self.filters,),
            initializer="ones",
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


def PadConv2D(
    x,
    conv_type="gen_conv",
    kernel_size=3,
    name=None,
    kernel_initializer="glorot_uniform",
    **kwargs,
):
    if kernel_size >= 3:
        padding_name = Base.format_name(name, "padding")
        x = tf.keras.layers.ZeroPadding2D(
            padding=(kernel_size // 2, kernel_size // 2), name=padding_name
        )(x)

    # Normal Conv2D
    if conv_type == "gen_conv":
        x = tf.keras.layers.Conv2D(
            kernel_size=kernel_size,
            padding="valid",
            name=name,
            kernel_initializer=kernel_initializer,
            **kwargs,
        )(x)

    # NFNet-style Conv2D
    elif conv_type == "nf_conv":
        w_init = tf.keras.initializers.VarianceScaling(1.0, "fan_in", "normal")
        x = ScaledWSConv2D(
            kernel_size=kernel_size,
            padding="valid",
            name=name,
            kernel_initializer=w_init,
            **kwargs,
        )(x)

    return x
