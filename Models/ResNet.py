import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base, CNNAttention, ConvLayers


def ResBlock(
    x,
    filters=64,
    strides=1,
    cnn_attention=None,
    stochdepth_rate=0.0,
    prefix=None,
):
    in_channels = x.shape[-1]

    out = x

    if strides > 1 or in_channels != filters * 4:
        if strides > 1:
            shortcut = tf.keras.layers.AveragePooling2D(
                padding="same",
                name=Base.format_name(prefix, "averagepooling2d_shortcut"),
            )(out)
        else:
            shortcut = out
        shortcut = ConvLayers.PadConv2D(
            shortcut,
            filters=filters * 4,
            kernel_size=1,
            conv_type="gen_conv",
            name=Base.format_name(prefix, "conv2d_shortcut"),
        )
    else:
        shortcut = x

    out = ConvLayers.PadConv2D(
        out,
        filters=filters,
        kernel_size=1,
        conv_type="gen_conv",
        name=Base.format_name(prefix, "conv2d_01"),
    )
    out = tf.keras.layers.BatchNormalization(name=Base.format_name(prefix, "bn_01"))(
        out
    )
    out = tf.keras.layers.ReLU(name=Base.format_name(prefix, "relu_01"))(out)

    out = ConvLayers.PadConv2D(
        out,
        filters=filters,
        kernel_size=3,
        strides=strides,
        conv_type="gen_conv",
        name=Base.format_name(prefix, "conv2d_02"),
    )
    out = tf.keras.layers.BatchNormalization(name=Base.format_name(prefix, "bn_02"))(
        out
    )
    out = tf.keras.layers.ReLU(name=Base.format_name(prefix, "relu_02"))(out)

    out = ConvLayers.PadConv2D(
        out,
        filters=filters * 4,
        kernel_size=1,
        conv_type="gen_conv",
        name=Base.format_name(prefix, "conv2d_03"),
    )
    out = tf.keras.layers.BatchNormalization(name=Base.format_name(prefix, "bn_03"))(
        out
    )

    if cnn_attention == "se":
        out = CNNAttention.SEBlock(filters * 4)(out)
    elif cnn_attention == "eca":
        out = CNNAttention.ECABlock(filters * 4)(out)

    if stochdepth_rate > 0.0:
        out = Base.StochDepth(drop_rate=stochdepth_rate, scale_by_keep=True)(out)

    out = Base.SkipInit()(out)

    out = tf.keras.layers.Add()([out, shortcut])
    out = tf.keras.layers.ReLU(name=Base.format_name(prefix, "relu_03"))(out)
    return out


definitions = {
    "50": {"blocks": [3, 4, 6, 3], "filters": [64, 128, 256, 512]},
    "101": {"blocks": [3, 4, 23, 3], "filters": [64, 128, 256, 512]},
    "152": {"blocks": [3, 8, 36, 3], "filters": [64, 128, 256, 512]},
}


def ResNetV1(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="50",
    cnn_attention=None,
):
    stochdepth_rate = 0.1
    definition = definitions[definition_name]

    num_blocks = sum(definition["blocks"])

    img_input = tf.keras.layers.Input(shape=in_shape)

    # Root block / "stem"
    x = ConvLayers.PadConv2D(
        img_input,
        filters=64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        conv_type="gen_conv",
        name="root_conv2d_01",
    )
    x = tf.keras.layers.BatchNormalization(name="root_bn_01")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name="root_maxpooling2d_01_pad")(
        x
    )
    x = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="valid", name="root_maxpooling2d_01"
    )(x)

    index = 0
    full_index = 0
    for stage_depth, block_width in zip(definition["blocks"], definition["filters"]):
        for block_index in range(stage_depth):
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            x = ResBlock(
                x,
                block_width,
                strides=2 if (block_index == 0 and index > 0) else 1,
                cnn_attention=cnn_attention,
                stochdepth_rate=block_stochdepth_rate,
                prefix="block%d_cell%d" % (index, block_index),
            )
            full_index += 1
        index += 1

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    dense_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    x = tf.keras.layers.Dense(
        out_classes, kernel_initializer=dense_init, name="predictions_dense"
    )(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="ResNet%sV1" % definition_name)
    return model
