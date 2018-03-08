#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jack on 06/03/2018

"""vocabulary generation.
    根据指定文件生成vocab文件
    统计词频截取高词频单词作为vocab
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import argparse

from tensor2tensor.data_generators import text_encoder

import tensorflow as tf


EOS = text_encoder.EOS

def read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " %s " % EOS).split()
    else:
      return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()

def build_vocab(sources, vocab_path, vocab_size):
  """Reads a file to build a vocabulary of `vocab_size` most common words.

   The vocabulary is sorted by occurrence count and has one word per line.
   Originally from:
   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py

  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
    vocab_size: size of the vocablulary to generate.
  """
  counter = collections.Counter()
  for filename in sources:
    data = read_words(filename)
    counter.update(collections.Counter(data))
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  words = words[:vocab_size]

  with open(vocab_path, "w") as f:
      for word in words:
          f.write(word+'\n')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='根据本地文件生成 vocabulary 文件')
    arg_parser.add_argument('-s', '--sources', dest='sources', metavar='local_sources_paths', required=True, nargs='*',
                            help='用于生成词表的文件路径')
    arg_parser.add_argument('-d', '--data_dir', dest='data_dir', help='词表文件所要存在的目录', required=True)
    arg_parser.add_argument('-size', '--vocab_size', dest='vocab_size', default=2**15, help='指定词表大小')
    arg_parser.add_argument('-vocab_name', '--vocab_name', dest='vocab_name', default='vocab.bpe', help='指定词表名称')

    args = arg_parser.parse_args()
    vocab_filename = args.vocab_name + '.' + str(args.vocab_size)
    vocab_filename = os.path.join(args.data_dir, vocab_filename)
    build_vocab(args.sources, vocab_filename, args.vocab_size)
    print("vocabulary file was built to be %s" % os.path.abspath(vocab_filename))
