#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jack on 06/03/2018

"""vocabulary generation.
    根据指定文件生成vocab文件
    利用系统的get_or_generate生成vocab文件，其中使用了SubwordTextEncoder
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
import argparse
import tensorflow as tf
import os
FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


def get_or_generate_local(data_dir, vocab_filename, vocab_size,
                          local_sources, file_byte_budget=1e6):
  """Generate a vocabulary from the datasets in sources."""

  def generate():
    tf.logging.info("Generating vocab from local: %s", str(local_sources))
    for source in local_sources:
      # Use Tokenizer to count the word occurrences.
      with tf.gfile.GFile(source, mode="r") as source_file:
        file_byte_budget_ = file_byte_budget
        counter = 0
        countermax = int(source_file.size() / file_byte_budget_ / 2)
        for line in source_file:
          if counter < countermax:
            counter += 1
          else:
            if file_byte_budget_ <= 0:
              break
            line = line.strip()
            file_byte_budget_ -= len(line)
            counter = 0
            yield line

  return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     generate())


def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator):
  """Inner implementation for vocab generators.

  Args:
    data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
    vocab_filename: relative filename where vocab file is stored
    vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
    generator: a generator that produces tokens from the vocabulary

  Returns:
    A SubwordTextEncoder vocabulary object.
  """
  if data_dir is None:
    vocab_filepath = None
  else:
    vocab_filepath = os.path.join(data_dir, vocab_filename)

  if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
    print("Found vocab file: %s", vocab_filepath)
    vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
    return vocab

  print("Generating vocab file: %s", vocab_filepath)
  token_counts = generator_utils.defaultdict(int)
  for item in generator:
    for tok in generator_utils.tokenizer.encode(text_encoder.native_to_unicode(item)):
      token_counts[tok] += 1

  vocab = text_encoder.SubwordTextEncoder.build_to_target_size(
      vocab_size, token_counts, 1, 1e3)


  if vocab_filepath is not None:
    vocab.store_to_file(vocab_filepath, False)
  return vocab


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='根据本地文件生成 vocabulary 文件')
    arg_parser.add_argument('-s', '--sources', dest='sources', metavar='local_sources_paths',required=True, nargs='*',
                            help='用于生成词表的文件路径')
    arg_parser.add_argument('-d', '--data_dir', dest='data_dir', help='词表文件所要存在的目录', required=True)
    arg_parser.add_argument('-size', '--vocab_size', dest='vocab_size', default=32000, help='指定词表大小')
    arg_parser.add_argument('-vocab_name', '--vocab_name', dest='vocab_name', default='vocab.bpe', help='指定词表名称')

    args = arg_parser.parse_args()
    vocab_filename = args.vocab_name+'.'+str(args.vocab_size)
    token_path = get_or_generate_local(
        args.data_dir, vocab_filename, args.vocab_size,
        args.sources)

    print("vocabulary file was built to be %s" % os.path.abspath(vocab_filename))