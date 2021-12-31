from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


def formatName(prefix, name):
    return (
        ("%s_%s" % (prefix, name)) if prefix is not None and name is not None else None
    )


def HeConv2D(x, filters=64, kernel_size=(3, 3), strides=(1, 1), name=None):
    if kernel_size >= 3:
        padding_name = ("%s_padding" % name) if name is not None else name
        x = layers.ZeroPadding2D(
            padding=(int(kernel_size / 2), int(kernel_size / 2)), name=padding_name
        )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(0.00005),
        name=name,
    )(x)
    return x


def ResBlockV2(x, filters=64, first=False, prefix=None):
    x1 = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_01")
    )(x)
    x1 = layers.ReLU(name=formatName(prefix, "relu_01"))(x1)

    x2 = x1

    x1 = HeConv2D(x1, filters, kernel_size=1, name=formatName(prefix, "conv2d_01"))

    x1 = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_02")
    )(x1)
    x1 = layers.ReLU(name=formatName(prefix, "relu_02"))(x1)
    x1 = HeConv2D(x1, filters, kernel_size=3, name=formatName(prefix, "conv2d_02"))

    x1 = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_03")
    )(x1)
    x1 = layers.ReLU(name=formatName(prefix, "relu_03"))(x1)
    x1 = HeConv2D(x1, filters * 4, kernel_size=1, name=formatName(prefix, "conv2d_03"))

    if first:
        x2 = HeConv2D(
            x2, filters * 4, kernel_size=1, name=formatName(prefix, "conv2d_shortcut")
        )
        x = x2 + x1
    else:
        x = x + x1
    return x


def DownBlockV2(x, filters=64, prefix=None):
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_01")
    )(x)
    x = layers.ReLU(name=formatName(prefix, "relu_01"))(x)

    x2 = x

    x1 = HeConv2D(x, filters, kernel_size=1, name=formatName(prefix, "conv2d_01"))

    x1 = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_02")
    )(x1)
    x1 = layers.ReLU(name=formatName(prefix, "relu_02"))(x1)
    x1 = HeConv2D(
        x1, filters, kernel_size=3, strides=2, name=formatName(prefix, "conv2d_02")
    )

    x1 = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name=formatName(prefix, "batchnorm_03")
    )(x1)
    x1 = layers.ReLU(name=formatName(prefix, "relu_03"))(x1)
    x1 = HeConv2D(x1, filters * 4, kernel_size=1, name=formatName(prefix, "conv2d_03"))

    x2 = layers.AveragePooling2D(
        padding="same", name=formatName(prefix, "averagepooling2d_shortcut")
    )(x2)
    x2 = HeConv2D(
        x2, filters * 4, kernel_size=1, name=formatName(prefix, "conv2d_shortcut")
    )

    x = x2 + x1
    return x


def ResNet50V4(in_shape=(320, 320, 3), out_classes=2000):
    img_input = layers.Input(shape=in_shape)

    # Root block / "stem"
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name="root_batchnorm_01"
    )(img_input)
    x = HeConv2D(x, filters=64, kernel_size=7, strides=2, name="root_conv2d_01")
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name="root_batchnorm_02"
    )(x)
    x = layers.ReLU(name="root_relu_01")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name="root_maxpooling2d_01_pad")(x)
    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="valid", name="root_maxpooling2d_01"
    )(x)

    # Block 1
    x = ResBlockV2(x, filters=64, first=True, prefix="block01_cell01")
    x = ResBlockV2(x, filters=64, prefix="block01_cell02")
    x = ResBlockV2(x, filters=64, prefix="block01_cell03")

    # Block 2
    x = DownBlockV2(x, filters=128, prefix="block02_cell01")
    x = ResBlockV2(x, filters=128, prefix="block02_cell02")
    x = ResBlockV2(x, filters=128, prefix="block02_cell03")
    x = ResBlockV2(x, filters=128, prefix="block02_cell04")

    # Block 3
    x = DownBlockV2(x, filters=256, prefix="block03_cell01")
    x = ResBlockV2(x, filters=256, prefix="block03_cell02")
    x = ResBlockV2(x, filters=256, prefix="block03_cell03")
    x = ResBlockV2(x, filters=256, prefix="block03_cell04")
    x = ResBlockV2(x, filters=256, prefix="block03_cell05")
    x = ResBlockV2(x, filters=256, prefix="block03_cell06")

    # Block 4
    x = DownBlockV2(x, filters=512, prefix="block04_cell01")
    x = ResBlockV2(x, filters=512, prefix="block04_cell02")
    x = ResBlockV2(x, filters=512, prefix="block04_cell03")

    # Classification block
    x = layers.BatchNormalization(
        momentum=0.9, epsilon=0.00001, axis=3, name="predictions_batchnorm"
    )(x)
    x = layers.ReLU(name="predictions_relu")(x)
    x = layers.GlobalAveragePooling2D(name="predictions_globalavgpooling")(x)

    x = layers.Dense(
        out_classes, kernel_initializer="he_normal", name="predictions_dense"
    )(x)
    x = layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = Model(img_input, x, name="ResNet50V4")
    return model
