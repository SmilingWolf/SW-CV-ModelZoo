import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base, CNNAttention, ConvLayers


def NFBlock(
    x,
    out_filters=64,
    alpha=1.0,
    beta=1.0,
    strides=1,
    group_size=128,
    expansion=0.5,
    cnn_attention=None,
    stochdepth_rate=0.0,
    compensate_avgpool_var=False,
    prefix=None,
    activation=Base.scaled_acts["relu"],
):
    in_channels = x.shape[-1]

    in_filters = int(out_filters * expansion)
    groups = in_filters // group_size
    in_filters = groups * group_size

    out = activation(x) * beta

    if strides > 1 or in_channels != out_filters:
        if strides > 1:
            shortcut = tf.keras.layers.AveragePooling2D(
                padding="same",
                name=Base.format_name(prefix, "averagepooling2d_shortcut"),
            )(out)

            # In the paper, authors note a k x k avg pooling operation
            # reduces variance by k, but note it doesn't affect training.
            # Here k = 2, as TF 2.7 default pooling window is 2x2
            if compensate_avgpool_var:
                shortcut = shortcut * 2
        else:
            shortcut = out
        shortcut = ConvLayers.PadConv2D(
            shortcut,
            filters=out_filters,
            kernel_size=1,
            conv_type="nf_conv",
            name=Base.format_name(prefix, "conv2d_shortcut"),
        )
    else:
        shortcut = x

    out = ConvLayers.PadConv2D(
        out,
        filters=in_filters,
        kernel_size=1,
        conv_type="nf_conv",
        name=Base.format_name(prefix, "conv2d_01"),
    )

    out = activation(out)
    out = ConvLayers.PadConv2D(
        out,
        filters=in_filters,
        kernel_size=3,
        strides=strides,
        groups=groups,
        conv_type="nf_conv",
        name=Base.format_name(prefix, "conv2d_02"),
    )

    out = activation(out)
    out = ConvLayers.PadConv2D(
        out,
        filters=in_filters,
        kernel_size=3,
        groups=groups,
        conv_type="nf_conv",
        name=Base.format_name(prefix, "conv2d_03"),
    )

    out = activation(out)
    out = ConvLayers.PadConv2D(
        out,
        filters=out_filters,
        kernel_size=1,
        conv_type="nf_conv",
        name=Base.format_name(prefix, "conv2d_04"),
    )

    if cnn_attention == "se":
        out = 2 * CNNAttention.SEBlock(out_filters)(out)  # Multiply by 2 for rescaling
    elif cnn_attention == "eca":
        out = 2 * CNNAttention.ECABlock(out_filters)(out)  # Multiply by 2 for rescaling

    if stochdepth_rate > 0.0:
        out = Base.StochDepth(drop_rate=stochdepth_rate)(out)

    out = Base.SkipInit()(out)

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
    "L2": {
        "blocks": [3, 6, 18, 9],
        "filters": [256, 512, 1536, 1536],
        "group_size": 64,
        "drop_rate": 0.4,
        "bneck_expansion": 0.25,
        "final_expansion": 2,
    },
}


def NFNetV1(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="L0",
    cnn_attention=None,
    compensate_avgpool_var=False,
    activation="relu",
    input_scaling="inception",
):
    alpha = 0.2
    width = 1.0
    stochdepth_rate = 0.1
    activation = Base.scaled_acts[activation]

    definition = definitions[definition_name]
    strides = [1, 2, 2, 2]

    num_blocks = sum(definition["blocks"])

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    # Root block / "stem"
    ch = definition["filters"][0] // 2
    x = ConvLayers.PadConv2D(
        x,
        filters=16,
        kernel_size=3,
        strides=2,
        conv_type="nf_conv",
        name="root_conv2d_01",
    )
    x = activation(x)

    x = ConvLayers.PadConv2D(
        x,
        filters=32,
        kernel_size=3,
        strides=1,
        conv_type="nf_conv",
        name="root_conv2d_02",
    )
    x = activation(x)

    x = ConvLayers.PadConv2D(
        x,
        filters=64,
        kernel_size=3,
        strides=1,
        conv_type="nf_conv",
        name="root_conv2d_03",
    )
    x = activation(x)

    x = ConvLayers.PadConv2D(
        x,
        filters=ch,
        kernel_size=3,
        strides=2,
        conv_type="nf_conv",
        name="root_conv2d_04",
    )

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
                cnn_attention=cnn_attention,
                stochdepth_rate=block_stochdepth_rate,
                compensate_avgpool_var=compensate_avgpool_var,
                prefix="block%d_cell%d" % (index, block_index),
                activation=activation,
            )

            ch = out_ch
            if block_index == 0:
                expected_std = 1.0
            expected_std = np.sqrt(expected_std**2 + alpha**2)
            full_index += 1
        index += 1

    # Classification block
    x = ConvLayers.PadConv2D(
        x,
        filters=int(ch * definition["final_expansion"]),
        kernel_size=1,
        conv_type="nf_conv",
        name="predictions_conv2d",
    )
    x = activation(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dropout(definition["drop_rate"])(x)

    dense_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    x = tf.keras.layers.Dense(
        out_classes, kernel_initializer=dense_init, name="predictions_dense"
    )(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="NFNet%sV1" % definition_name)
    return model
