import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base


def MLPBlock(x, mlp_dim, prefix=""):
    out = tf.keras.layers.Dense(mlp_dim, name=f"{prefix}_dense_01")(x)
    out = tf.keras.layers.Activation(activation="gelu", name=f"{prefix}_act_01")(out)
    out = tf.keras.layers.Dense(x.shape[-1], name=f"{prefix}_dense_02")(out)
    return out


def MixerBlock(x, tokens_mlp_dim, channels_mlp_dim, prefix=""):
    out = x

    out = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm_01")(out)
    out = tf.keras.layers.Permute(dims=(2, 1))(out)
    out = MLPBlock(out, tokens_mlp_dim, prefix=f"{prefix}_tm")
    out = tf.keras.layers.Permute(dims=(2, 1))(out)
    x = tf.keras.layers.Add()([out, x])

    out = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm_02")(x)
    out = MLPBlock(out, channels_mlp_dim, prefix=f"{prefix}_cm")
    out = tf.keras.layers.Add()([out, x])
    return out


definitions = {
    "S_32": {
        "num_blocks": 8,
        "patch_size": 32,
        "hidden_dim": 512,
        "tokens_mlp_dim": 256,
        "channels_mlp_dim": 2048,
    },
    "S_16": {
        "num_blocks": 8,
        "patch_size": 16,
        "hidden_dim": 512,
        "tokens_mlp_dim": 256,
        "channels_mlp_dim": 2048,
    },
    "B_32": {
        "num_blocks": 12,
        "patch_size": 32,
        "hidden_dim": 768,
        "tokens_mlp_dim": 384,
        "channels_mlp_dim": 3072,
    },
    "B_16": {
        "num_blocks": 12,
        "patch_size": 16,
        "hidden_dim": 768,
        "tokens_mlp_dim": 384,
        "channels_mlp_dim": 3072,
    },
    "L_32": {
        "num_blocks": 24,
        "patch_size": 32,
        "hidden_dim": 1024,
        "tokens_mlp_dim": 512,
        "channels_mlp_dim": 4096,
    },
    "L_16": {
        "num_blocks": 24,
        "patch_size": 16,
        "hidden_dim": 1024,
        "tokens_mlp_dim": 512,
        "channels_mlp_dim": 4096,
    },
    "H_14": {
        "num_blocks": 32,
        "patch_size": 14,
        "hidden_dim": 1280,
        "tokens_mlp_dim": 640,
        "channels_mlp_dim": 5120,
    },
}


def MLPMixer(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="S_16",
    input_scaling="inception",
):
    definition = definitions[definition_name]
    num_blocks = definition["num_blocks"]
    patch_size = definition["patch_size"]
    hidden_dim = definition["hidden_dim"]
    tokens_mlp_dim = definition["tokens_mlp_dim"]
    channels_mlp_dim = definition["channels_mlp_dim"]

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    prefix = "root"
    x = tf.keras.layers.Conv2D(
        filters=hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name=f"{prefix}_conv2d_01",
    )(x)
    x = tf.keras.layers.Reshape(target_shape=(-1, hidden_dim))(x)

    for i in range(num_blocks):
        prefix = f"block{i}"
        x = MixerBlock(x, tokens_mlp_dim, channels_mlp_dim, prefix)

    x = tf.keras.layers.LayerNormalization(name="predictions_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dense(out_classes, name="predictions_dense")(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"MLPMixer-{definition_name}")
    return model
