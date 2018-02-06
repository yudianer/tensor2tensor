#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jack on 04/02/2018

# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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


@registry.register_problem("translate_mnzh_by_32kbpe")
class TranslateMnzh32k(translate.TranslateProblem):
  """
    Problem spec for Mn_unicode-zh translation.
  """
  @property
  def targeted_vocab_size(self):
    return 32000  # 32k

  @property
  def source_vocab_name(self):
    return "vocab.32k.mn.txt"

  @property
  def target_vocab_name(self):
    return "vocab.32k.ch.txt"


  def generator(self, data_dir, tmp_dir, train):
    source_vocab = os.path.join(data_dir,self.source_vocab_name)
    target_vocab = os.path.join(data_dir,self.target_vocab_name)
    tag = "train" if train else "dev"

    filename_base = "%s.32k" % tag
    """Concatenate all `datasets` and save to `filename`.   return tmp_dir/filename_base """

    data_path = os.path.join(data_dir,filename_base)
    """Generator for sequence-to-sequence tasks that uses tokens.

      This generator assumes the files at source_path and target_path have
      the same number of lines and yields dictionaries of "inputs" and "targets"
      where inputs are token ids from the " "-split source (and target, resp.) lines
      converted to integers using the token_map.

      Args:
        source_path: path to the file with source sentences.
        target_path: path to the file with target sentences.
        source_token_vocab: text_encoder.TextEncoder object.
        target_token_vocab: text_encoder.TextEncoder object.
        eos: integer to append at the end of each sequence (default: None).
      Yields:
        A dictionary {"inputs": source-line, "targets": target-line} where
        the lines are integer lists converted from tokens in the file lines.
      """
    source_token_vocab = text_encoder.TokenTextEncoder(source_vocab, replace_oov="UNK")
    target_token_vocab = text_encoder.TokenTextEncoder(target_vocab, replace_oov="UNK")
    return translate.token_generator_by_source_target(data_path + ".mn",
                                               data_path + ".ch",
                                                      source_token_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_TOK

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.TokenTextEncoder(source_vocab_filename)
    target_token = text_encoder.TokenTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }
