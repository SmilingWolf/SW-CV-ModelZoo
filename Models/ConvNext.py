import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base, CNNAttention


class Affine(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.alpha = self.add_weight(
            name="alpha",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return self.alpha * x + self.beta


class GlobalResponseNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        channels = input_shape[-1]
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, channels),
            initializer="zeros",
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, channels),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        gx = tf.norm(x, ord="euclidean", axis=(1, 2), keepdims=True)
        nx = gx / (tf.math.reduce_mean(gx, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


def ResBlock(
    x,
    prefix,
    filters=64,
    cnn_attention=None,
    stochdepth_rate=0.0,
    norm_func=tf.keras.layers.LayerNormalization,
    version=1,
):
    out = x

    out = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(out)
    out = tf.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        padding="valid",
        name=f"{prefix}_conv2d_01",
    )(out)
    out = norm_func(name=f"{prefix}_norm_01")(out)

    out = tf.keras.layers.Conv2D(
        filters=filters * 4, kernel_size=1, padding="same", name=f"{prefix}_conv2d_02"
    )(out)
    out = tf.keras.layers.Activation(activation="gelu", name=f"{prefix}_act_02")(out)

    if version == 2:
        out = GlobalResponseNorm(name=f"{prefix}_grn")(out)

    out = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, padding="same", name=f"{prefix}_conv2d_03"
    )(out)

    if cnn_attention == "se":
        out = CNNAttention.SEBlock(filters)(out)
    elif cnn_attention == "eca":
        out = CNNAttention.ECABlock(filters)(out)

    if stochdepth_rate > 0.0:
        out = Base.StochDepth(drop_rate=stochdepth_rate, scale_by_keep=True)(out)

    if version == 1:
        out = Base.SkipInitChannelwise(filters)(out)

    out = tf.keras.layers.Add()([out, x])
    return out


definitions = {
    "T": {"blocks": [3, 3, 9, 3], "filters": [96, 192, 384, 768]},
    "S": {"blocks": [3, 3, 27, 3], "filters": [96, 192, 384, 768]},
    "B": {"blocks": [3, 3, 27, 3], "filters": [128, 256, 512, 1024]},
    "L": {"blocks": [3, 3, 27, 3], "filters": [192, 384, 768, 1536]},
}

model_norms = {
    "batchnorm": tf.keras.layers.BatchNormalization,
    "layernorm": tf.keras.layers.LayerNormalization,
    "affine": Affine,
}


def ConvNext(
    in_shape,
    out_classes,
    definition_name,
    cnn_attention,
    input_scaling,
    stochdepth_rate,
    norm,
    version,
):
    definition = definitions[definition_name]
    norm_func = model_norms[norm]

    num_blocks = sum(definition["blocks"])

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    # Root block / "stem"
    x = tf.keras.layers.Conv2D(
        filters=definition["filters"][0],
        kernel_size=4,
        strides=4,
        use_bias=True,
        padding="same",
        name="root_conv2d",
    )(x)
    x = norm_func(name="root_norm")(x)

    index = 0
    full_index = 0
    for stage_depth, block_width in zip(definition["blocks"], definition["filters"]):
        if index > 0:
            x = norm_func(name="block%d_down_norm" % index)(x)
            x = tf.keras.layers.Conv2D(
                filters=block_width,
                kernel_size=2,
                strides=2,
                use_bias=True,
                padding="same",
                name=f"block{index}_down_conv2d",
            )(x)

        for block_index in range(stage_depth):
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            x = ResBlock(
                x,
                prefix=f"block{index}_cell{block_index}",
                filters=block_width,
                cnn_attention=cnn_attention,
                stochdepth_rate=block_stochdepth_rate,
                norm_func=norm_func,
                version=version,
            )
            full_index += 1
        index += 1

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = norm_func(name="predictions_norm")(x)

    dense_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    x = tf.keras.layers.Dense(
        out_classes, kernel_initializer=dense_init, name="predictions_dense"
    )(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"ConvNext{definition_name}V{version}")
    return model


def ConvNextV1(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="S",
    cnn_attention=None,
    input_scaling="inception",
    stochdepth_rate=0.1,
    norm="layernorm",
):
    return ConvNext(
        in_shape=in_shape,
        out_classes=out_classes,
        definition_name=definition_name,
        cnn_attention=cnn_attention,
        input_scaling=input_scaling,
        stochdepth_rate=stochdepth_rate,
        norm=norm,
        version=1,
    )


def ConvNextV2(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="S",
    cnn_attention=None,
    input_scaling="inception",
    stochdepth_rate=0.1,
    norm="layernorm",
):
    return ConvNext(
        in_shape=in_shape,
        out_classes=out_classes,
        definition_name=definition_name,
        cnn_attention=cnn_attention,
        input_scaling=input_scaling,
        stochdepth_rate=stochdepth_rate,
        norm=norm,
        version=2,
    )
