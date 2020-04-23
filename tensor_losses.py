
__all__ = ["tensor_mse", "tensor_nse", "tensor_r2", "tensor_kge"]

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def np_kge(true, predicted):
    """
    Kling-Gupta Efficiency
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance
     criteria: Implications for improving hydrological modelling
    output:
        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    cc = np.corrcoef(true, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(true)
    beta = np.sum(predicted) / np.sum(true)
    _kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return _kge.astype(np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.float32)])
def tensor_kge(input1, input2):
    y = tf.numpy_function(np_kge, [input1, input2], tf.float32)
    return y


def covariance(true, predicted):
    """
    Covariance
        .. math::
        Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
    """
    obs_mean = np.mean(true)
    sim_mean = np.mean(predicted)
    covar = np.mean((true - obs_mean)*(predicted - sim_mean))
    return covar


def np_nse(true, predicted):
    _nse = 1 - sum((predicted - true) ** 2) / sum((true - np.mean(true)) ** 2)
    return _nse


def tensor_nse(true, pred, name='nse'):
    neum = tf.reduce_sum(tf.square(tf.subtract(pred, true)))
    denom = tf.reduce_sum(tf.square(tf.subtract(true, tf.math.reduce_mean(true))))
    const = tf.constant(1.0, dtype=tf.float32)
    return tf.subtract(const, tf.math.divide(neum, denom), name=name)


def tensor_r2(pred, true, name):
    """
  R_squared computes the coefficient of determination.
  It is a measure of how well the observed outcomes are replicated by the model.
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(true, pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(true, tf.reduce_mean(true))))
    const = tf.constant(1.0, dtype=tf.float32)
    r2 = tf.subtract(const, tf.div(residual, total), name=name)
    return r2


tensor_mse = tf.keras.losses.MSE


if __name__ == "__main__":
    t, p = np.random.random(3), np.random.random(3)
    t_tensor = tf.constant(t, dtype=tf.float32)
    p_tensor = tf.constant(p, dtype=tf.float32)

    print(K.eval(tensor_kge(t_tensor, p_tensor), np_kge(t, p)))

    print(K.eval(tensor_nse(t_tensor, p_tensor)), np_nse(t, p))
