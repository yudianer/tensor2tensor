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
      problem_choice="adaptive",
      num_decoder_layers="6",
      hidden_size="512",
      sampling_method="argmax",
      kernel_width=1,
      attention_key_channels="0",
      max_input_seq_length=0,
      moe_k=2,
      learning_rate_warmup_steps=16000,
      use_pad_remover=True,
      num_hidden_layers=6,
      num_encoder_layers="6",
      moe_hidden_sizes="2048",
      compress_steps=0,
      no_data_parallelism=False,
      max_length="256",
      multiply_embedding_mode="sqrt_depth",
      relu_dropout_broadcast_dims="",
      norm_type="layer",
      layer_postprocess_sequence="da",
      force_full_predict=False,
      learning_rate_minimum=None,
      scheduled_sampling_gold_mixin_prob=0.5,
      ffn_layer="dense_relu_dense",
      sampling_temp=1.0,
      optimizer_adam_beta2=0.997,
      optimizer="Adam",
      min_length=0,
      optimizer_adam_beta1=0.9,
      shared_embedding_and_softmax_weights="False",
      summarize_grads=False,
      proximity_bias=False,
      learning_rate_cosine_cycle_steps=250000,
      symbol_modality_skip_top=False,
      scheduled_sampling_prob=0.0,
      factored_logits=False,
      learning_rate_decay_steps=5000,
      parameter_attention_value_channels=0,
      initializer="uniform_unit_scaling",
      parameter_attention_key_channels=0,
      use_fixed_batch_size=False,
      optimizer_adam_epsilon=1e-09,
      nbr_decoder_problems=1,
      summarize_vars=False,
      min_length_bucket=8,
      max_target_seq_length=0,
      moe_loss_coef=0.01,
      length_bucket_step=1.1,
      clip_grad_norm="0.0",
      learning_rate="1.0",
      weight_decay=0.0,
      grad_noise_scale=0.0,
      attention_dropout_broadcast_dims="",
      learning_rate_decay_scheme="noam",
      layer_prepostprocess_dropout=0.1,
      pos="timing",
      attention_dropout="0.0",
      symbol_dropout=0.0,
      prepend_mode="none",
      learning_rate_decay_staircase=False,
      max_relative_position=0,
      layer_preprocess_sequence="n",
      initializer_gain="1.0",
      norm_epsilon=1e-06,
      dropout=0.2,
      daisy_chain_variables=True,
      eval_drop_long_sequences=False,
      moe_num_experts=64,
      symbol_modality_num_shards=16,
      filter_size="2048",
      scheduled_sampling_warmup_steps=50000,
      label_smoothing="0.1",
      relu_dropout="0.0",
      layer_prepostprocess_dropout_broadcast_dims="",
      kernel_height=3,
      batch_size="6250",
      input_modalities="default",
      eval_run_autoregressive=False,
      self_attention_type="dot_product",
      optimizer_momentum_nesterov=False,
      split_to_length=0,
      optimizer_momentum_momentum=0.9,
      learning_rate_decay_rate=1.0,
      num_heads="8",
      weight_noise=0.0,
      attention_value_channels="0",
      target_modality="default",
  )