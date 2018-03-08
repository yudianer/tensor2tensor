#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jack on 04/02/2018

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

@registry.register_problem
class TranslateMnzhBpe32k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version.

    此表和句子都是使用的经过BPE处理之后的文件。
    自己产生vocab文件，源语言和目标语言共用一个vocab文件
    不知如何使用多个校验集？
  """

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32000

  @property
  def vocab_name(self):
    return "vocab.bpe"

  @property
  def source_vocab_name(self):
    return "vocab.32k.mn.txt"

  @property
  def target_vocab_name(self):
      return "vocab.32k.ch.txt"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")
    return {"inputs": encoder, "targets": encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the mn->zh task, training set."""
    dataset_path = ("train.32k"
                    if train else "valid.32k")
    train_path = os.path.join(data_dir, dataset_path)

    vocab = generator_utils.get_or_generate_local(
      data_dir, self.vocab_file, self.targeted_vocab_size,
      ['train.32k.mn', 'train.32k.ch', 'valid.32k.mn', 'valid.32k.ch'])

    token_path = os.path.join(data_dir, self.vocab_file)
    with tf.gfile.GFile(token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(token_path, mode="w") as f:
      f.write(vocab_data)

    token_vocab = text_encoder.TokenTextEncoder(token_path, replace_oov="UNK")
    return translate.token_generator(train_path + ".mn", train_path + ".ch",
                                     token_vocab, EOS)
  @property
  def input_space_id(self):
    return problem.SpaceID.MN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_BPE_TOK

@registry.register_problem
class TranslateMnzh32k(translate.TranslateProblem):
  """Problem spec for Mn-Zh translation.
    withou BPE

    自己产生vocab文件，源文件和目标文件使用同一个vocab文件
  """

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_name(self):
    return "vocab.ende"


  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the mn->zh task, training set."""
    dataset_path = ("train.32k"
                    if train else "valid.32k")
    symbolizer_vocab = generator_utils.get_or_generate_local(
      data_dir, self.vocab_file, self.targeted_vocab_size,
      ['train.32k.mn', 'train.32k.ch', 'valid.32k.mn', 'valid.32k.ch'])

    train_path = os.path.join(data_dir, dataset_path)

    return translate.token_generator(train_path + ".mn", train_path + ".ch",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_TOK


@registry.register_problem
class TranslateMnzhBpeSt32k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version.

    此表和句子都是使用的经过BPE处理之后的文件。
    自己不构建vocab文件，源语言和目标语言使用不同的vocab文件
    不知如何使用多个校验集？
  """

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe"

  @property
  def source_vocab_name(self):
    return "vocab.32k.mn.txt"

  @property
  def target_vocab_name(self):
      return "vocab.32k.ch.txt"

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="UNK")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="UNK")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the mn->zh task, training set."""
    dataset_path = ("train.32k"
                    if train else "valid.32k")
    train_path = os.path.join(data_dir, dataset_path)

    source_token_path = os.path.join(data_dir, self.source_vocab_name)
    target_token_path = os.path.join(data_dir, self.target_vocab_name)
    with tf.gfile.GFile(source_token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(source_token_path, mode="w") as f:
      f.write(vocab_data)

    with tf.gfile.GFile(target_token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(target_token_path, mode="w") as f:
      f.write(vocab_data)

    source_token_vocab = text_encoder.TokenTextEncoder(source_token_path, replace_oov="UNK")
    target_token_vocab = text_encoder.TokenTextEncoder(target_token_path, replace_oov="UNK")
    return translate.token_generator_by_source_target(train_path + ".mn", train_path + ".ch",
                                     source_token_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_BPE_TOK


@registry.register_problem
class TranslateMnzhBpe32kBak(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version.

    此表和句子都是使用的经过BPE处理之后的文件。
    不知如何使用多个校验集？
  """

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe"

  @property
  def source_vocab_name(self):
    return "vocab.32k.mn.txt"

  @property
  def target_vocab_name(self):
      return "vocab.32k.ch.txt"

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="UNK")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="UNK")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the mn->zh task, training set."""
    dataset_path = ("train.32k"
                    if train else "valid.32k")
    train_path = os.path.join(data_dir, dataset_path)

    source_token_path = os.path.join(data_dir, self.source_vocab_name)
    target_token_path = os.path.join(data_dir, self.target_vocab_name)
    with tf.gfile.GFile(source_token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(source_token_path, mode="w") as f:
      f.write(vocab_data)

    with tf.gfile.GFile(target_token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(target_token_path, mode="w") as f:
      f.write(vocab_data)

    source_token_vocab = text_encoder.TokenTextEncoder(source_token_path, replace_oov="UNK")
    target_token_vocab = text_encoder.TokenTextEncoder(target_token_path, replace_oov="UNK")
    return translate.token_generator_by_source_target(train_path + ".mn", train_path + ".ch",
                                     source_token_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_BPE_TOK
