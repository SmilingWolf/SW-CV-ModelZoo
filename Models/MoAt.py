import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base, MoAtAttention


class MoAtWindower(tf.keras.layers.Layer):
    def __init__(self, window_height, window_width, **kwargs):
        super().__init__(**kwargs)
        self.window_height = window_height
        self.window_width = window_width

    def call(self, inputs, training):
        _, height, width, channels = inputs.shape.with_rank(4).as_list()
        inputs = tf.reshape(
            inputs,
            (
                -1,
                height // self.window_height,
                self.window_height,
                width // self.window_width,
                self.window_width,
                channels,
            ),
        )
        inputs = tf.transpose(inputs, (0, 1, 3, 2, 4, 5))
        inputs = tf.reshape(
            inputs, (-1, self.window_height, self.window_width, channels)
        )
        return inputs


class MoAtUnwindower(tf.keras.layers.Layer):
    def __init__(self, height, width, window_height, window_width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.window_height = window_height
        self.window_width = window_width

    def call(self, inputs, training):
        _, _, channels = inputs.shape.with_rank(3).as_list()
        inputs = tf.reshape(
            inputs,
            [
                -1,
                self.height // self.window_height,
                self.width // self.window_width,
                self.window_height,
                self.window_width,
                channels,
            ],
        )
        inputs = tf.transpose(inputs, (0, 1, 3, 2, 4, 5))
        inputs = tf.reshape(inputs, (-1, self.height, self.width, channels))
        return inputs


def SEBlock(x, se_filters, prefix=""):
    c = x.shape[-1]

    attn = tf.keras.layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)
    attn = tf.keras.layers.Dense(units=se_filters, name=f"{prefix}_dense_01")(attn)
    attn = tf.keras.layers.Activation("gelu", name=f"{prefix}_act_01")(attn)
    attn = tf.keras.layers.Dense(units=c, name=f"{prefix}_dense_02")(attn)
    attn = tf.keras.layers.Activation("sigmoid", name=f"{prefix}_act_02")(attn)
    x = tf.keras.layers.Multiply(name=f"{prefix}_scaling")([x, attn])
    return x


def MBConvBlock(x, mb_dim, strides, se_ratio, stochdepth_rate, prefix=""):
    c = x.shape[-1]
    se_filters = mb_dim * se_ratio

    shortcut = x

    if strides > 1:
        shortcut = tf.keras.layers.AveragePooling2D(
            pool_size=3, strides=2, padding="same", name=f"{prefix}_shortcut_avgpool"
        )(shortcut)

    if c != mb_dim:
        shortcut = tf.keras.layers.Conv2D(
            filters=mb_dim, kernel_size=1, name=f"{prefix}_shortcut_conv"
        )(shortcut)

    out = tf.keras.layers.experimental.SyncBatchNormalization(name=f"{prefix}_bn_01")(x)
    out = tf.keras.layers.Conv2D(
        filters=mb_dim * 4,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=f"{prefix}_conv2d_01",
    )(out)
    out = tf.keras.layers.experimental.SyncBatchNormalization(name=f"{prefix}_bn_02")(
        out
    )
    out = tf.keras.layers.Activation("gelu", name=f"{prefix}_act_01")(out)
    out = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{prefix}_conv2d_02",
    )(out)
    out = tf.keras.layers.experimental.SyncBatchNormalization(
        gamma_initializer="zeros", name=f"{prefix}_bn_03"
    )(out)
    out = tf.keras.layers.Activation("gelu", name=f"{prefix}_act_02")(out)

    if se_ratio > 0:
        out = SEBlock(out, se_filters, prefix=f"{prefix}_se")

    out = tf.keras.layers.Conv2D(
        filters=mb_dim, kernel_size=1, padding="same", name=f"{prefix}_conv2d_03"
    )(out)

    if stochdepth_rate > 0:
        out = Base.StochDepth(
            drop_rate=stochdepth_rate, scale_by_keep=True, name=f"{prefix}_sd"
        )(out)

    x = tf.keras.layers.Add(name=f"{prefix}_add")([shortcut, out])
    return x


def MoAtBlock(
    x,
    block_type,
    filters,
    strides,
    se_ratio,
    head_size,
    window_side,
    stochdepth_rate,
    use_pe,
    prefix="",
):
    x = MBConvBlock(
        x, filters, strides, se_ratio, stochdepth_rate, prefix=f"{prefix}_mb"
    )

    if block_type == "moat":
        _, h, w, c = x.shape

        out = tf.keras.layers.LayerNormalization(name=f"{prefix}_ln")(x)

        if window_side is not None:
            out = MoAtWindower(
                window_height=window_side,
                window_width=window_side,
                name=f"{prefix}_window",
            )(out)

        out = MoAtAttention.MoAtAttention(
            hidden_size=filters,
            head_size=head_size,
            relative_position_embedding_type="2d_multi_head" if use_pe else None,
            name=f"{prefix}_attention",
        )(out)

        if window_side is not None:
            out = MoAtUnwindower(
                height=h,
                width=w,
                window_height=window_side,
                window_width=window_side,
                name=f"{prefix}_unwindow",
            )(out)
        else:
            out = tf.keras.layers.Reshape((h, w, c), name=f"{prefix}_reshape")(out)

        if stochdepth_rate > 0:
            out = Base.StochDepth(
                drop_rate=stochdepth_rate, scale_by_keep=True, name=f"{prefix}_sd"
            )(out)

        x = tf.keras.layers.Add(name=f"{prefix}_add")([out, x])
    return x


def MoAtStem(x, stem_filters):
    prefix = "root"
    x = tf.keras.layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=True,
        name=f"{prefix}_conv2d_01",
    )(x)
    x = tf.keras.layers.experimental.SyncBatchNormalization(name=f"{prefix}_bn")(x)
    x = tf.keras.layers.Activation("gelu", name=f"{prefix}_act")(x)
    x = tf.keras.layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"{prefix}_conv2d_02",
    )(x)
    return x


definitions = {
    "MoAt0": {
        "stem_filters": 64,
        "blocks": [2, 3, 7, 2],
        "block_types": ["mb", "mb", "moat", "moat"],
        "filters": [96, 192, 384, 768],
        "stochdepth_rate": 0.1,
    },
    "MoAt1": {
        "stem_filters": 64,
        "blocks": [2, 6, 14, 2],
        "block_types": ["mb", "mb", "moat", "moat"],
        "filters": [96, 192, 384, 768],
        "stochdepth_rate": 0.2,
    },
    "MoAt2": {
        "stem_filters": 128,
        "blocks": [2, 6, 14, 2],
        "block_types": ["mb", "mb", "moat", "moat"],
        "filters": [128, 256, 512, 1024],
        "stochdepth_rate": 0.3,
    },
    "MoAt3": {
        "stem_filters": 160,
        "blocks": [2, 12, 28, 2],
        "block_types": ["mb", "mb", "moat", "moat"],
        "filters": [160, 320, 640, 1280],
        "stochdepth_rate": 0.6,
    },
}


def MoAt(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="MoAt2",
    input_scaling="inception",
    window_sides=[None, None, 20, None],
    use_pe=False,
    **kwargs,
):
    definition = definitions[definition_name]
    stem_filters = definition["stem_filters"]
    blocks = definition["blocks"]
    block_types = definition["block_types"]
    filters = definition["filters"]
    stochdepth_rate = kwargs.get("stochdepth_rate", definition["stochdepth_rate"])

    num_blocks = sum(definition["blocks"])

    head_size = 32

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    x = MoAtStem(x, stem_filters)

    full_index = 0
    for (i, stage), block_type, mb_dim, window_side in zip(
        enumerate(blocks), block_types, filters, window_sides
    ):
        for j in range(stage):
            prefix = f"stage{i}_block{j}"
            strides = 2 if j == 0 else 1
            se_ratio = 0.25 if block_type == "mb" else 0.0
            block_stochdepth_rate = stochdepth_rate * full_index / num_blocks
            x = MoAtBlock(
                x,
                block_type,
                mb_dim,
                strides,
                se_ratio,
                head_size,
                window_side,
                block_stochdepth_rate,
                use_pe,
                prefix,
            )
            full_index += 1

    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.LayerNormalization(name="predictions_norm")(x)
    x = tf.keras.layers.Dense(out_classes, name="predictions_dense")(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"{definition_name}")
    return model
