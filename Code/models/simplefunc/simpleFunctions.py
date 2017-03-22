#-*- coding: utf-8 -*-

import tensorflow as tf

class simpleFunc(object):
  def __call__(self, dataset, moving_params=None):
    inputs = dataset.inputs
    self.token_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs, 0)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.token_to_keep3D, [1, 2]), [-1, 1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
		

    output = {}
    output["token_to_keep3D"] = self.token_to_keep3D
    output["sequence_lengths"] = self.sequence_lengths
    output["n_tokens"] = self.n_tokens
    return output