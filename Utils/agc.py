""" An implementation of Adaptive Gradient Clipping
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
Code references:
  * Official JAX implementation (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
  * Ross Wightman's implementation https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
"""

import tensorflow as tf


def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [
            0,
            1,
            2,
        ]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01, eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        if "predictions_dense" not in params.name:
            p_norm = unitwise_norm(params)
            max_norm = tf.math.maximum(p_norm, eps) * clip_factor
            grad_norm = unitwise_norm(grads)
            clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
            new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        else:
            new_grad = grads
        new_grads.append(new_grad)
    return new_grads
