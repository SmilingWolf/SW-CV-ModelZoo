import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base


class Affine(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
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


def MLP(x, dim, prefix=""):
    out = tf.keras.layers.Dense(dim * 4, name=f"{prefix}_dense_01")(x)
    out = tf.keras.layers.Activation(activation="gelu", name=f"{prefix}_act_01")(out)
    out = tf.keras.layers.Dense(dim, name=f"{prefix}_dense_02")(out)
    return out


def CaiT_LayerScale_init(network_depth):
    if network_depth <= 18:
        return 1e-1
    elif network_depth <= 24:
        return 1e-5
    else:
        return 1e-6


def ResMLP_Blocks(x, nb_patches, dim, layerscale_init, prefix=""):
    out = x
    out = Affine(channels=dim)(out)
    out = tf.keras.layers.Permute(dims=(2, 1))(out)
    out = tf.keras.layers.Dense(nb_patches, name=f"{prefix}_dense_01")(out)
    out = tf.keras.layers.Permute(dims=(2, 1))(out)
    out = Base.SkipInitChannelwise(channels=dim, init_val=layerscale_init)(out)
    x = tf.keras.layers.Add()([out, x])

    out = x
    out = Affine(channels=dim)(out)
    out = MLP(out, dim, prefix=f"{prefix}_mlp")
    out = Base.SkipInitChannelwise(channels=dim, init_val=layerscale_init)(out)
    out = tf.keras.layers.Add()([out, x])
    return out


definitions = {
    "RMLP-S12": {"patch_size": 16, "dim": 384, "depth": 12},
    "RMLP-S24": {"patch_size": 16, "dim": 384, "depth": 24},
    "RMLP-B24": {"patch_size": 16, "dim": 768, "depth": 24},
}


def ResMLP(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="RMLP-S24",
    input_scaling="inception",
):
    definition = definitions[definition_name]
    dim = definition["dim"]
    depth = definition["depth"]
    patch_size = definition["patch_size"]
    layerscale_init = CaiT_LayerScale_init(depth)

    nb_patches = (in_shape[0] * in_shape[1]) // (patch_size**2)

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    prefix = "root"
    x = tf.keras.layers.Conv2D(
        filters=dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name=f"{prefix}_conv2d_01",
    )(x)
    x = tf.keras.layers.Reshape(target_shape=(-1, dim))(x)

    for i in range(depth):
        prefix = f"block{i}"
        x = ResMLP_Blocks(x, nb_patches, dim, layerscale_init, prefix)

    x = Affine(dim)(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dense(out_classes, name="predictions_dense")(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"ResMLP-{definition_name}")
    return model
