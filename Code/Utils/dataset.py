# -*- coding: utf-8 -*-



import numpy as np
import tensorflow as tf

from sentence import Sentence
from vocab import Vocab


class Dataset(object):

  def __init__(self, filename, vocabs):
    self.vocabs = vocabs
    self.filename = filename
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='inputs')
    self.buff = []
    self.batch_size = 2
    self.n_sents = 0
    self.build()

  def build(self):
    with open(self.filename) as f:
      for line in f:
        line = line.strip().split()
        if line:
          self.buff.append(self._process_sent(line))
          self.n_sents += 1

  def _process_sent(self, line):
    wids = []
    sent = Sentence()
    for word in line:
      wids.append(self.vocabs[0][word])
    sent.words = line
    sent.wids = np.array(wids)
    return sent

  def get_batch(self):
    n_sents = 0
    while n_sents < self.n_sents:
      data = []
      words = []
      maxlen = 0
      for i in xrange(n_sents, min(n_sents+self.batch_size, self.n_sents)):
        if len(self.buff[i].wids) > maxlen:
          maxlen = len(self.buff[i].wids)
        data.append(self.buff[i].wids)
        words.append(self.buff[i].words)
      shape = (self.batch_size, maxlen)
      fdata = np.zeros(shape, dtype=np.int32)
      for i, datum in enumerate(data):
        datum = np.array(datum)
        fdata[i, 0:len(datum)] = datum
      feed_dict = {self.inputs : fdata}
      n_sents += self.batch_size
      yield feed_dict, words
