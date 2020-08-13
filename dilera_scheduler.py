from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
import sys

@keras_export("keras.optimizers.schedules.Dilera")
class Dilera(LearningRateSchedule):

  def __init__(
      self,
      initial_learning_rate,
      sigma,
      seed, 
      name=None):

    super(Dilera, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.sigma = sigma
    self.seed = seed
    self.name = name
    tf.random.set_seed(seed)

  def __call__(self, step):
    with tf.name_scope(self.name or "Dilera") as name:
      dtype = tf.dtypes.float32

      initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, dtype=dtype, name="initial_learning_rate")

      sigma = math_ops.cast(self.sigma, dtype)
      t_step = math_ops.cast(step, dtype)
      # t_step = math_ops.multiply(t_step, t_step)
      dt = tf.constant(1, dtype=dtype)
      t_step = math_ops.add(t_step, tf.constant(1, dtype=dtype))
      Z_t = tf.random.normal([1], mean=0.0, stddev=1.0, dtype=dtype)

      Z_over_T = math_ops.divide(Z_t[0], t_step)
      Sigma_Z_over_T = math_ops.multiply(sigma, Z_over_T)
      Sigma_Z_sqrtDt_over_T = math_ops.multiply(Sigma_Z_over_T, math_ops.sqrt(dt))

      eta_dT = math_ops.multiply(initial_learning_rate, dt)
      newLearningRate  = math_ops.subtract(eta_dT, Sigma_Z_sqrtDt_over_T, name=name)
      return newLearningRate

  def get_config(self):
      
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "sigma": self.sigma,
        "seed": self.seed,
        "name": self.name
    }
