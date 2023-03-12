"""Surrogate Gap Sharpness Aware Minimization implementation.
   Heavily based on the Keras implementation of SAM [1]
   and JAX implementation of GSAM [2].

   Main difference from Keras SAM: epsilon is increased from 1e-12 to 1e-7
   to improve stability with mixed precision training.

   Reference:
     [1] https://github.com/keras-team/keras/blob/v2.10.0/keras/models/sharpness_aware_minimization.py
     [2] https://github.com/google-research/big_vision/blob/c62890a3e4487b1d6751794b090138b9da5d18e1/big_vision/trainers/proj/gsam/gsam.py
"""

import copy

import tensorflow as tf
from keras.engine import data_adapter
from keras.layers import deserialize as deserialize_layer
from keras.models import Model
from keras.utils import generic_utils


class GapSharpnessAwareMinimization(Model):
    def __init__(
        self,
        model,
        rho_max=0.05,
        rho_min=0.05,
        alpha=0.0,
        lr_max=0.0,
        lr_min=0.0,
        num_batch_splits=None,
        name=None,
    ):
        super().__init__(name=name)
        self.model = model
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.alpha = alpha
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.num_batch_splits = num_batch_splits
        self.eps = 1e-7

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.lr_max == self.lr_min:
            rho = self.rho_max
        else:
            rho = self.rho_min + (self.rho_max - self.rho_min) * (
                self.optimizer._decayed_lr(tf.float32) - self.lr_min
            ) / (self.lr_max - self.lr_min)

        if self.num_batch_splits is not None:
            x_split = tf.split(x, self.num_batch_splits)
            y_split = tf.split(y, self.num_batch_splits)
        else:
            x_split = [x]
            y_split = [y]

        gradients_all_batches = []
        pred_all_batches = []
        for x_batch, y_batch in zip(x_split, y_split):
            epsilon_w_cache = []
            with tf.GradientTape() as tape:
                pred = self.model(x_batch)
                loss = self.compiled_loss(y_batch, pred)
            pred_all_batches.append(pred)
            trainable_variables = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)

            gradients_order2_norm = self._gradients_order2_norm(gradients)
            scale = rho / (gradients_order2_norm + self.eps)

            for gradient, variable in zip(gradients, trainable_variables):
                epsilon_w = gradient * scale
                self._distributed_apply_epsilon_w(
                    variable, epsilon_w, tf.distribute.get_strategy()
                )
                epsilon_w_cache.append(epsilon_w)

            with tf.GradientTape() as tape:
                pred = self(x_batch)
                loss = self.compiled_loss(y_batch, pred)
            g_robust = tape.gradient(loss, trainable_variables)

            # --- GSAM starts here
            g_robust_order2_norm = self._gradients_order2_norm(g_robust)
            g_robust_normalized = [x / g_robust_order2_norm for x in g_robust]
            g_clean_projection_norm = tf.math.reduce_sum(
                [
                    tf.experimental.numpy.vdot(p, q)
                    for p, q in zip(g_robust_normalized, gradients)
                ]
            )
            g_clean_residual = [
                (a - g_clean_projection_norm * b)
                for a, b in zip(gradients, g_robust_normalized)
            ]
            g_gsam = [(a - b * self.alpha) for a, b in zip(g_robust, g_clean_residual)]
            # --- GSAM ends here

            if len(gradients_all_batches) == 0:
                for gradient in g_gsam:
                    gradients_all_batches.append([gradient])
            else:
                for gradient, gradient_all_batches in zip(
                    g_gsam, gradients_all_batches
                ):
                    gradient_all_batches.append(gradient)
            for variable, epsilon_w in zip(trainable_variables, epsilon_w_cache):
                # Restore the variable to its original value before
                # `apply_gradients()`.
                self._distributed_apply_epsilon_w(
                    variable, -epsilon_w, tf.distribute.get_strategy()
                )

        gradients = []
        for gradient_all_batches in gradients_all_batches:
            gradients.append(tf.math.reduce_sum(gradient_all_batches, axis=0))
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        pred = tf.concat(pred_all_batches, axis=0)
        self.compiled_metrics.update_state(y, pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": generic_utils.serialize_keras_object(self.model),
                "rho_max": self.rho_max,
                "rho_min": self.rho_min,
                "alpha": self.alpha,
                "lr_max": self.lr_max,
                "lr_min": self.lr_min,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Avoid mutating the input dict.
        config = copy.deepcopy(config)
        model = deserialize_layer(config.pop("model"), custom_objects=custom_objects)
        config["model"] = model
        return super().from_config(config, custom_objects)

    def _distributed_apply_epsilon_w(self, var, epsilon_w, strategy):
        # Helper function to apply epsilon_w on model variables.
        if isinstance(
            tf.distribute.get_strategy(),
            (
                tf.distribute.experimental.ParameterServerStrategy,
                tf.distribute.experimental.CentralStorageStrategy,
            ),
        ):
            # Under PSS and CSS, the AggregatingVariable has to be kept in sync.
            def distribute_apply(strategy, var, epsilon_w):
                strategy.extended.update(
                    var,
                    lambda x, y: x.assign_add(y),
                    args=(epsilon_w,),
                    group=False,
                )

            tf.__internal__.distribute.interim.maybe_merge_call(
                distribute_apply, tf.distribute.get_strategy(), var, epsilon_w
            )
        else:
            var.assign_add(epsilon_w)

    def _gradients_order2_norm(self, gradients):
        norm = tf.norm(
            tf.stack([tf.norm(grad) for grad in gradients if grad is not None])
        )
        return norm
