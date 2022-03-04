import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base, CNNAttention


def ResBlock(
    x,
    filters=64,
    cnn_attention=None,
    stochdepth_rate=0.0,
    prefix=None,
):
    out = x

    out = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(out)
    out = tf.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        padding="valid",
        name=Base.format_name(prefix, "conv2d_01"),
    )(out)
    out = tf.keras.layers.LayerNormalization(name=Base.format_name(prefix, "norm_01"))(
        out
    )

    out = tf.keras.layers.Conv2D(
        filters=filters * 4,
        kernel_size=1,
        padding="same",
        name=Base.format_name(prefix, "conv2d_02"),
    )(out)
    out = tf.keras.layers.Activation(
        activation="gelu", name=Base.format_name(prefix, "act_02")
    )(out)

    out = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        name=Base.format_name(prefix, "conv2d_03"),
    )(out)

    if cnn_attention == "se":
        out = CNNAttention.SEBlock(filters)(out)
    elif cnn_attention == "eca":
        out = CNNAttention.ECABlock(filters)(out)

    if stochdepth_rate > 0.0:
        out = Base.StochDepth(drop_rate=stochdepth_rate, scale_by_keep=True)(out)

    out = Base.SkipInitChannelwise(filters)(out)

    out = tf.keras.layers.Add()([out, x])
    return out


definitions = {
    "T": {"blocks": [3, 3, 9, 3], "filters": [96, 192, 384, 768]},
    "S": {"blocks": [3, 3, 27, 3], "filters": [96, 192, 384, 768]},
    "B": {"blocks": [3, 3, 27, 3], "filters": [128, 256, 512, 1024]},
    "L": {"blocks": [3, 3, 27, 3], "filters": [192, 384, 768, 1536]},
}


def ConvNextV1(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="S",
    cnn_attention=None,
    input_scaling="inception",
    stochdepth_rate=0.1,
):
    definition = definitions[definition_name]

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
    x = tf.keras.layers.LayerNormalization(name="root_norm")(x)

    index = 0
    full_index = 0
    for stage_depth, block_width in zip(definition["blocks"], definition["filters"]):
        if index > 0:
            x = tf.keras.layers.LayerNormalization(name="block%d_down_norm" % index)(x)
            x = tf.keras.layers.Conv2D(
                filters=block_width,
                kernel_size=2,
                strides=2,
                use_bias=True,
                padding="same",
                name="block%d_down_conv2d" % index,
            )(x)

        for block_index in range(stage_depth):
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            x = ResBlock(
                x,
                block_width,
                cnn_attention=cnn_attention,
                stochdepth_rate=block_stochdepth_rate,
                prefix="block%d_cell%d" % (index, block_index),
            )
            full_index += 1
        index += 1

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.LayerNormalization(name="predictions_norm")(x)

    dense_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    x = tf.keras.layers.Dense(
        out_classes, kernel_initializer=dense_init, name="predictions_dense"
    )(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="ConvNext%sV1" % definition_name)
    return model
