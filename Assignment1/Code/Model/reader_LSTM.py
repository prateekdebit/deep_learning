"""Utilities for parsing """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os,sys

import numpy as np


import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().split()

def _build_vocab_mine() :
	f = open('new_glove_3.txt', 'r')
	words = []
	vectors = []
	idx = -1
	count =0
	word_dict = {}
	word_to_id = {}
	word_dict_reverse = {}
	for l in f:
		idx += 1
		line = l
		strings = line.split()
		if len(strings)==301:
			word = strings[0]
			vector = [np.float32(num) for num in strings[1:]]
			if len(vector)==300:
				word_dict[word] = idx
				word_dict_reverse[idx] = word
				words.append(word)
				vector = np.array(vector).reshape((1,-1))
				vectors.append(vector)
		else :
			count+=1
	f.close()

	word_to_id  = word_dict
	return word_to_id







def _build_vocab(filename):

  word_to_id = {}
  
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


def model_raw_data(data_path=None):


  train_path = os.path.join(data_path, "ptb.train.txt")
  test_path1 = os.path.join(data_path, "test1.txt")
  test_path2 = os.path.join(data_path, "test2.txt")
  test_path3 = os.path.join(data_path, "test3.txt")
  test_path4 = os.path.join(data_path, "test4.txt")
  test_path5 = os.path.join(data_path, "test5.txt")

  word_to_id = _build_vocab_mine()
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(train_path, word_to_id)
  test_data1 = _file_to_word_ids(test_path1, word_to_id)
  test_data2 = _file_to_word_ids(test_path2, word_to_id)
  test_data3 = _file_to_word_ids(test_path3, word_to_id)
  test_data4 = _file_to_word_ids(test_path4, word_to_id)
  test_data5 = _file_to_word_ids(test_path5, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data1, test_data2, test_data3, test_data4, test_data5, vocabulary


def iterator(raw_data, batch_size, num_steps):

  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
