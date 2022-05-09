import tensorflow as tf
from tensorflow.keras.models import Model

from .layers import Base


def MixerBlock(x, filters, kernel_size=7, activation="relu", prefix=""):
    out = x
    out = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size, padding="same", name=f"{prefix}_conv2d_01"
    )(out)
    out = tf.keras.layers.Activation(activation=activation, name=f"{prefix}_act_01")(
        out
    )
    out = tf.keras.layers.BatchNormalization(
        gamma_initializer="zeros", name=f"{prefix}_bn_01"
    )(out)
    out = tf.keras.layers.Add()([out, x])

    out = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        name=f"{prefix}_conv2d_02",
    )(out)
    out = tf.keras.layers.Activation(activation=activation, name=f"{prefix}_act_02")(
        out
    )
    out = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn_02")(out)
    return out


definitions = {
    "CM-1536_20": {
        "filters": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
        "activation": "gelu",
    },
    "CM-768_32": {
        "filters": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
        "activation": "relu",
    },
    "CMSW-1024_20": {
        "filters": 1024,
        "depth": 20,
        "patch_size": 16,
        "kernel_size": 9,
        "activation": "gelu",
    },
}


def ConvMixer(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="CMSW-1024_20",
    input_scaling="inception",
):
    definition = definitions[definition_name]
    filters = definition["filters"]
    depth = definition["depth"]
    patch_size = definition["patch_size"]
    kernel_size = definition["kernel_size"]
    activation = definition["activation"]

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    prefix = "root"
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name=f"{prefix}_conv2d_01",
    )(x)
    x = tf.keras.layers.Activation(activation=activation, name=f"{prefix}_act_01")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn_01")(x)

    for i in range(depth):
        prefix = f"block{i}"
        x = MixerBlock(x, filters, kernel_size, activation, prefix)

    x = tf.keras.layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)
    x = tf.keras.layers.Dense(out_classes, name="predictions_dense")(x)
    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name=f"ConvMixer-{definition_name}")
    return model
