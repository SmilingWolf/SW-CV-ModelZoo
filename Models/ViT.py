import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base


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
        r = tf.random.uniform(shape=[batch_size, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        config.update({"scale_by_keep": self.scale_by_keep})
        return config


class PosEmbed(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        emb_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(input_shape[-2], input_shape[-1]),
            initializer=emb_init,
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embed


def MLPBlock(x, mlp_dim, stochdepth_rate, prefix=""):
    out = tf.keras.layers.Dense(mlp_dim, name=f"{prefix}_dense_01")(x)
    out = tf.keras.layers.Activation(activation="gelu", name=f"{prefix}_act_01")(out)
    if stochdepth_rate > 0.0:
        out = StochDepth(stochdepth_rate, scale_by_keep=True)(out)

    out = tf.keras.layers.Dense(x.shape[-1], name=f"{prefix}_dense_02")(out)
    return out


def ViTBlock(x, heads, key_dim, mlp_dim, layerscale_init, stochdepth_rate, prefix=""):
    out = x

    out = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm_01")(out)
    out = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim // heads)(
        out, out
    )
    out = Base.SkipInitChannelwise(channels=key_dim, init_val=layerscale_init)(out)

    if stochdepth_rate > 0.0:
        out = StochDepth(stochdepth_rate, scale_by_keep=True)(out)

    x = tf.keras.layers.Add()([out, x])

    out = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm_02")(x)
    out = MLPBlock(out, mlp_dim, stochdepth_rate, prefix=f"{prefix}_cm")
    out = Base.SkipInitChannelwise(channels=key_dim, init_val=layerscale_init)(out)

    if stochdepth_rate > 0.0:
        out = StochDepth(stochdepth_rate, scale_by_keep=True)(out)

    out = tf.keras.layers.Add()([out, x])
    return out


def CaiT_LayerScale_init(network_depth):
    if network_depth <= 18:
        return 1e-1
    elif network_depth <= 24:
        return 1e-5
    else:
        return 1e-6


definitions = {
    "B16": {
        "num_blocks": 12,
        "patch_size": 16,
        "key_dim": 768,
        "mlp_dim": 3072,
        "heads": 12,
        "stochdepth_rate": 0.05,
    },
    "B32": {
        "num_blocks": 12,
        "patch_size": 32,
        "key_dim": 768,
        "mlp_dim": 3072,
        "heads": 12,
        "stochdepth_rate": 0.05,
    },
    "L16": {
        "num_blocks": 24,
        "patch_size": 16,
        "key_dim": 1024,
        "mlp_dim": 4096,
        "heads": 16,
        "stochdepth_rate": 0.2,
    },
}


def ViT(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="B16",
    input_scaling="inception",
):
    definition = definitions[definition_name]
    num_blocks = definition["num_blocks"]
    patch_size = definition["patch_size"]
    key_dim = definition["key_dim"]
    mlp_dim = definition["mlp_dim"]
    heads = definition["heads"]
    stochdepth_rate = definition["stochdepth_rate"]
    layerscale_init = CaiT_LayerScale_init(num_blocks)

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    prefix = "root"
    x = tf.keras.layers.Conv2D(
        filters=key_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name=f"{prefix}_conv2d_01",
    )(x)
    x = tf.keras.layers.Reshape(target_shape=(-1, key_dim))(x)
    x = PosEmbed()(x)

    for i in range(num_blocks):
        prefix = f"block{i}"
        x = ViTBlock(
            x, heads, key_dim, mlp_dim, layerscale_init, stochdepth_rate, prefix
        )

    x = tf.keras.layers.LayerNormalization(name="predictions_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dense(out_classes, name="predictions_dense")(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"ViT-{definition_name}")
    return model
