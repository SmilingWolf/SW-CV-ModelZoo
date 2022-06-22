# Credits to Sayak Paul
# https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow/blob/main/SAM.ipynb

import tensorflow as tf


class SAMModel(tf.keras.Model):
    def __init__(self, inner_model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.inner_model = inner_model
        self.rho = rho

    @tf.function
    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.inner_model(images, training=True)
            loss = self.compiled_loss(labels, predictions)

        trainable_params = self.inner_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.inner_model(images)
            loss = self.compiled_loss(labels, predictions)

        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients(zip(sam_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        (images, labels) = data
        predictions = self.inner_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([tf.norm(grad) for grad in gradients if grad is not None])
        )
        return norm

    def save_weights(self, filepath, *args, **kwargs):
        self.inner_model.save(filepath=filepath)

    def call(self, inputs, *args, **kwargs):
        return self.inner_model(inputs, *args, **kwargs)
