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

"""Hyperparameters and ranges common to multiple models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.utils import registry

import tensorflow as tf
import json
import os

@registry.register_hparams("jack_basic_params")
def jack_basic_params():
  """
  A set of basic hyperparameters from jack_basic_params.json set by Jack
  """
  # params_file_path=os.path.split(os.path.realpath(__file__))[0] + '/jack_t2t_params.json'
  # hparams = \
  return tf.contrib.training.HParams(
      ffn_layer="dense_relu_dense",
      pos="timing",
      use_pad_remover=True,
      learning_rate_cosine_cycle_steps=250000,
      norm_type="layer",
      optimizer_momentum_momentum=0.9,
      num_hidden_layers=6,
      kernel_height=3,
      clip_grad_norm=0.0,
      daisy_chain_variables=True,
      prepend_mode="none",
      norm_epsilon=1e-06,
      num_decoder_layers=6,
      self_attention_type="dot_product",
      moe_k=2,
      learning_rate_decay_rate=1.0,
      problem_choice="adaptive",
      filter_size=2048,
      batch_size=6250,
      kernel_width=1,
      length_bucket_step=1.1,
      attention_value_channels=0,
      attention_dropout=0.0,
      learning_rate_warmup_steps=16000,
      nbr_decoder_problems=1,
      scheduled_sampling_gold_mixin_prob=0.5,
      sampling_temp=1.0,
      hidden_size=512,
      sampling_method="argmax",
      moe_hidden_sizes=2048,
      learning_rate=1.0,
      learning_rate_decay_steps=5000,
      num_encoder_layers=6,
      initializer_gain=1.0,
      layer_postprocess_sequence="da",
      initializer="uniform_unit_scaling",
      optimizer_adam_epsilon=1e-09,
      input_modalities="default",
      attention_key_channels=0,
      optimizer_adam_beta2=0.997,
      multiply_embedding_mode="sqrt_depth",
      scheduled_sampling_warmup_steps=50000,
      target_modality="default",
      label_smoothing=0.1,
      optimizer_adam_beta1=0.9,
      min_length_bucket=8,
      max_length=256,
      layer_preprocess_sequence="n",
      layer_prepostprocess_dropout=0.1,
      moe_num_experts=64,
      num_heads=8,
      optimizer="Adam",
      symbol_modality_num_shards=16,
      relu_dropout=0.0,
      dropout=0.2,
      learning_rate_decay_scheme="noam",
      moe_loss_coef=0.01,
      shared_embedding_and_softmax_weights=False

  )