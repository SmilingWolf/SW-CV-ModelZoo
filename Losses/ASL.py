import tensorflow as tf
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper


class AsymmetricLoss(LossFunctionWrapper):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        eps=1e-7,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="asymmetric_loss",
    ):
        super().__init__(
            asymmetric_loss,
            name=name,
            reduction=reduction,
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
        )


@tf.function
def asymmetric_loss(y_true, y_pred, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    xs_pos = y_pred
    xs_neg = 1 - y_pred

    if clip is not None and clip > 0:
        xs_neg = tf.clip_by_value(xs_neg + clip, clip_value_min=0, clip_value_max=1)

    # Basic CE calculation
    los_pos = y_true * tf.math.log(
        tf.clip_by_value(xs_pos, clip_value_min=eps, clip_value_max=1)
    )
    los_neg = (1 - y_true) * tf.math.log(
        tf.clip_by_value(xs_neg, clip_value_min=eps, clip_value_max=1)
    )
    loss = los_pos + los_neg

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y_true
        pt1 = xs_neg * (1 - y_true)
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return -tf.reduce_sum(loss, axis=-1)
