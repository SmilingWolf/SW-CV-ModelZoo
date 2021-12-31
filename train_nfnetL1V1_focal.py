import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.metrics import F1Score

from Generator.Upscale_DB import DataGenerator
from Models.NFNet import NFNetV1
from Utils import agc, mixup


def scheduler(epoch, lr):
    if epoch == 35:
        return lr * 0.1
    if epoch == 53:
        return lr * 0.1
    if epoch == 68:
        return lr * 0.1
    if epoch == 73:
        return lr * 0.1
    if epoch == 90:
        return lr * 0.1
    if epoch == 97:
        return lr * 0.1
    else:
        return lr


class AGCModel(tf.keras.Model):
    def __init__(self, inner_model, clip_factor=0.02, eps=1e-3):
        super(AGCModel, self).__init__()
        self.inner_model = inner_model
        self.clip_factor = clip_factor
        self.eps = eps

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            predictions = self.inner_model(images, training=True)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.inner_model.trainable_weights
        gradients = tape.gradient(loss, trainable_params)
        agc_gradients = agc.adaptive_clip_grad(
            trainable_params, gradients, clip_factor=self.clip_factor, eps=self.eps
        )
        self.optimizer.apply_gradients(zip(agc_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self.inner_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, filepath, *args, **kwargs):
        super(AGCModel, self).save_weights(filepath=filepath + "_train")
        self.inner_model.save(filepath=filepath)

    def call(self, inputs, *args, **kwargs):
        return self.inner_model(inputs)


if __name__ == "__main__":
    dim = 320
    miniBatch = 32
    f1 = F1Score(2380, "micro", 0.4)
    model = NFNetV1(
        in_shape=(dim, dim, 3), out_classes=2380, definition_name="L1", use_eca=False
    )
    model.load_weights(
        "checkpoints/NFNetL1V1-rmc-83-0.54438/NFNetL1V1-rmc-83-0.54438/variables/variables"
    )

    model = AGCModel(model)

    loss = SigmoidFocalCrossEntropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    opt = SGD(learning_rate=0.3, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=loss, metrics=[f1])

    arg = np.random.rand(miniBatch, dim, dim, 3)
    model.predict(arg)

    print("Number of parameters: %d" % model.count_params())

    trainList = open("2020_0000_0599/trainlist.txt", "r").readlines()
    trainList = [x.rstrip() for x in trainList]

    labels_list = pd.read_csv("2020_0000_0599/selected_tags.csv")["tag_id"].tolist()

    training_generator = DataGenerator(
        trainList, labels_list, noise_level=2, dim=(dim, dim)
    )

    training_dataset = training_generator.genDS()
    training_dataset = training_dataset.shuffle(len(trainList))
    training_dataset = training_dataset.map(
        training_generator.wrap_func, num_parallel_calls=tf.data.AUTOTUNE
    )
    training_dataset = training_dataset.batch(miniBatch, drop_remainder=True)

    training_dataset_mu = training_dataset.map(
        lambda images, labels: mixup.mix_up_single(images, labels, alpha=0.3),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    training_dataset_mu = training_dataset_mu.prefetch(tf.data.AUTOTUNE)

    sched = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=True)
    rmc = tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/NFNetL1V1-rmc-{epoch:02d}-{f1_score:.5f}/NFNetL1V1-rmc-{epoch:02d}-{f1_score:.5f}",
        save_best_only=False,
        save_freq="epoch",
        save_weights_only=True
    )

    model.fit(
        training_dataset_mu,
        validation_data=None,
        initial_epoch=83,
        epochs=100,
        callbacks=[sched, rmc],
    )
