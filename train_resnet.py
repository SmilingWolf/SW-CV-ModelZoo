from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb
from tensorflow_addons.metrics import F1Score
from tensorflow_addons.optimizers import LAMB
from wandb.keras import WandbCallback

from Generator.ParseTFRecord import DataGenerator
from Losses.ASL import AsymmetricLoss
from Models.ResNet import ResNetV1


def scheduler(epoch, lr):
    if epoch < warmup_epochs:
        linear_decay = (max_learning_rate - warmup_learning_rate) / warmup_epochs
        return warmup_learning_rate + linear_decay * epoch
    else:
        pi_decay = (epoch - warmup_epochs) / max((total_epochs - 1 - warmup_epochs), 1)
        cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * pi_decay))
        alpha = final_learning_rate / max_learning_rate
        decayed = (1 - alpha) * cosine_decay + alpha
        return max_learning_rate * decayed


if __name__ == "__main__":
    try:  # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:  # detect GPUs
        strategy = tf.distribute.MirroredStrategy()  # for CPU/GPU or multi-GPU machines

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%Hh%Mm%Ss")

    multiplier = 1
    node_name = "vm_name_here"
    bucket_root = "gs://sw_tpu_training"

    # Input
    image_size = 320
    total_labels = 2380
    global_batch_size = 32 * multiplier * strategy.num_replicas_in_sync

    # Training schedule
    warmup_epochs = 0
    total_epochs = 100

    # Learning rate
    max_learning_rate = 5e-4 * multiplier
    warmup_learning_rate = max_learning_rate * 0.1
    final_learning_rate = max_learning_rate * 0.01

    # Model definition
    definition_name = "50"
    cnn_attention = None
    activation = "relu"
    stochdepth_rate = 0.1
    dropout_rate = 0.0

    # Augmentations
    noise_level = 2
    mixup_alpha = 0.8
    cutout_max_pct = 0.0
    random_resize_method = True

    # Loss
    loss_name = "asl"
    asl_gamma_neg = 0
    asl_gamma_pos = 0
    asl_clip = 0.0

    train_config = {
        "image_size": image_size,
        "total_labels": total_labels,
        "global_batch_size": global_batch_size,
        "warmup_epochs": warmup_epochs,
        "total_epochs": total_epochs,
        "max_learning_rate": max_learning_rate,
        "warmup_learning_rate": warmup_learning_rate,
        "final_learning_rate": final_learning_rate,
        "definition_name": definition_name,
        "cnn_attention": cnn_attention,
        "activation": activation,
        "stochdepth_rate": stochdepth_rate,
        "dropout_rate": dropout_rate,
        "noise_level": noise_level,
        "mixup_alpha": mixup_alpha,
        "cutout_max_pct": cutout_max_pct,
        "random_resize_method": random_resize_method,
        "loss_name": loss_name,
        "asl_gamma_neg": asl_gamma_neg,
        "asl_gamma_pos": asl_gamma_pos,
        "asl_clip": asl_clip,
    }

    wandb_run = wandb.init(
        project="tpu-tracking",
        entity="smilingwolf",
        config=train_config,
        name="ResNet%sV1_%s" % (definition_name, date_time),
        tags=[node_name],
    )

    training_generator = DataGenerator(
        "%s/%s/record_shards_train/*" % (bucket_root, node_name),
        total_labels=total_labels,
        image_size=image_size,
        batch_size=global_batch_size,
        noise_level=noise_level,
        mixup_alpha=mixup_alpha,
        cutout_max_pct=cutout_max_pct,
        random_resize_method=random_resize_method,
    )
    training_dataset = training_generator.genDS()

    validation_generator = DataGenerator(
        "%s/%s/record_shards_val/*" % (bucket_root, node_name),
        total_labels=total_labels,
        image_size=image_size,
        batch_size=global_batch_size,
        noise_level=0,
        mixup_alpha=0.0,
        cutout_max_pct=0.0,
        random_resize_method=False,
    )
    validation_dataset = validation_generator.genDS()

    with strategy.scope():
        model = ResNetV1(
            in_shape=(image_size, image_size, 3),
            out_classes=total_labels,
            definition_name=definition_name,
            cnn_attention=cnn_attention,
            input_scaling="inception",
            stochdepth_rate=stochdepth_rate,
            dropout_rate=dropout_rate,
        )

        f1 = F1Score(total_labels, "micro", 0.4)
        rec_at_p65 = tf.keras.metrics.RecallAtPrecision(0.65, num_thresholds=1024)
        loss = AsymmetricLoss(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
        )
        opt = LAMB(learning_rate=warmup_learning_rate, weight_decay_rate=0.05)
        model.compile(optimizer=opt, loss=loss, metrics=[f1, rec_at_p65])

    t800 = tf.keras.callbacks.TerminateOnNaN()
    sched = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=True)
    rmc_loss = tf.keras.callbacks.ModelCheckpoint(
        "%s/checkpoints/ResNet%sV1_%s/variables/variables"
        % (bucket_root, definition_name, date_time),
        save_best_only=True,
        save_freq="epoch",
        save_weights_only=True,
    )
    cb_list = [t800, rmc_loss, sched, WandbCallback(save_model=False)]

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        initial_epoch=0,
        epochs=total_epochs,
        steps_per_epoch=10996 // multiplier,
        validation_steps=364 // multiplier,
        callbacks=cb_list,
    )

    wandb_run.finish()
