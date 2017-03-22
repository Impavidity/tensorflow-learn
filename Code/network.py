#-*- coding: utf8 -*-
from Utils.vocab import Vocab
from Utils.dataset import Dataset
import models
import tensorflow as tf



class Network(object):
  def __init__(self, model):
    self.train_file = "../Data/text.in"
    self._vocabs = []
    self._model = model()
    self._vocabs.append(Vocab(self.train_file))
    self._trainset = Dataset(self.train_file, self._vocabs)
    self._ops = self._gen_ops()

  def train_minibatch(self):
    return self._trainset.get_batch()

  def _gen_ops(self):
    train_output = self._model(self._trainset)
    ops = {}
    ops['train_op'] = [train_output['token_to_keep3D'],
                       train_output['sequence_lengths'],
                       train_output['n_tokens']]
    return ops

  def train(self, sess):
    for feed_dict, words in self.train_minibatch():
      print words
      print feed_dict
      token_to_keep3D, sequence_lengths, n_tokens = sess.run(self.ops['train_op'], feed_dict=feed_dict)
      print "token_to_keep3D", token_to_keep3D
      print "sequence_lengths", sequence_lengths
      print "n_tokens", n_tokens
      print "n_tokens_from_original_data", sum([len(word) for word in words])

  @property
  def ops(self):
    return self._ops

  

if __name__=="__main__":
  model = getattr(models, "simpleFunc")
  network = Network(model)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    network.train(sess)

