import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.metrics import F1Score

from Generator.Upscale_DB import DataGenerator
from Models.ResNet import ResNet50V4


def scheduler(epoch, lr):
    if epoch == 15:
        return lr * 0.1
    if epoch == 26:
        return lr * 0.1
    else:
        return lr


if __name__ == "__main__":
    dim = 320
    miniBatch = 64
    f1 = F1Score(2380, "micro", 0.4)
    model = ResNet50V4(in_shape=(dim, dim, 3), out_classes=2380)

    loss = SigmoidFocalCrossEntropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    opt = SGD(learning_rate=0.025, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=loss, metrics=[f1])

    resumeModel = "ResNet50V4-rmc-12-0.52158"
    if resumeModel != "":
        model.load_weights(
            "checkpoints/%s/%s/variables/variables" % (resumeModel, resumeModel)
        )
        K.set_value(model.optimizer.lr, 0.05)

    print("Number of parameters: %d" % model.count_params())

    trainList = open("2020_0000_0599/trainlist.txt", "r").readlines()
    trainList = [x.rstrip() for x in trainList]

    labels_list = pd.read_csv("2020_0000_0599/selected_tags.csv")["tag_id"].tolist()

    training_generator = DataGenerator(
        trainList, labels_list, noise_level=2, dim=(dim, dim), batch_size=miniBatch
    )

    sched = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=True)
    rmc = tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/ResNet50V4-rmc-{epoch:02d}-{f1_score:.5f}/ResNet50V4-rmc-{epoch:02d}-{f1_score:.5f}",
        save_best_only=False,
        save_freq="epoch",
    )

    model.fit_generator(
        generator=training_generator,
        validation_data=None,
        initial_epoch=12,
        epochs=30,
        use_multiprocessing=False,
        workers=8,
        callbacks=[sched, rmc],
    )
