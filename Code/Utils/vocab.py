# -*- coding: utf-8 -*-

class Vocab(object):
  SPECIAL_TOKENS = ('<PAD>','<ROOT>','<UNK>')
  START_IDX = len(SPECIAL_TOKENS)
  PAD, ROOT, UNK = range(START_IDX)

  def __init__(self, vocab_file):
    self._vocab_file = vocab_file
    self._str2id = dict(zip(Vocab.SPECIAL_TOKENS, range(Vocab.START_IDX)))
    self._id2str = dict(zip(range(Vocab.START_IDX), Vocab.SPECIAL_TOKENS))
    self._cur_idx = Vocab.START_IDX
    self.add_train_file()

  def process(self, line):
    for word in line:
      if word not in self._str2id:
        self._str2id[word] = self._cur_idx
        self._id2str[self._cur_idx] = word
        self._cur_idx += 1


  def add_train_file(self):
    with open(self._vocab_file) as f:
      for line_num, line in enumerate(f):
        line = line.strip().split()
        if line:
          self.process(line)

  def __getitem__(self, key):
    if isinstance(key, basestring):
      return self._str2id.get(key, Vocab.UNK)
    else:
      raise ValueError("The key is not supported!\n")
 