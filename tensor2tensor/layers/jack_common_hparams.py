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


@registry.register_hparams("jack_common_hparams")
def jack_common_hparams():
  """A set of basic hyperparameters set by Jack """
  return tf.contrib.training.HParams(
      kernel_height=3,
      pos="timing",
      sampling_temp=1.0,
      num_encoder_layers=6,
      learning_rate=0.1,
      daisy_chain_variables=True,
      num_hidden_layers=6,
      self_attention_type="dot_product",
      eval_run_autoregressive=False,
      compress_steps=0,
      proximity_bias=False,
      moe_k=2,
      clip_grad_norm=0.0,
      layer_postprocess_sequence="layer_norm",
      learning_rate_decay_scheme="noam",
      scheduled_sampling_prob=0.0,
      weight_noise=0.0,
      initializer_gain=1.0,
      shared_embedding_and_softmax_weights=False,
      use_fixed_batch_size=False,
      prepend_mode="none",
      optimizer_adam_beta2=0.997,
      moe_loss_coef=0.01,
      layer_prepostprocess_dropout_broadcast_dims="",
      optimizer="Adam",
      initializer="uniform_unit_scaling",
      hidden_size=512,
      force_full_predict=False,
      learning_rate_decay_rate=1.0,
      symbol_modality_skip_top=False,
      sampling_method="argmax",
      summarize_vars=False,
      num_decoder_layers=6,
      grad_noise_scale=0.0,
      max_target_seq_length=0,
      max_length=256,
      split_to_length=0,
      parameter_attention_key_channels=0,
      symbol_modality_num_shards=16,
      attention_key_channels=0,
      layer_prepostprocess_dropout=0.1,
      input_modalities="default",
      label_smoothing=0.1,
      learning_rate_decay_steps=5000,
      learning_rate_minimum=None,
      layer_preprocess_sequence="n",
      learning_rate_decay_staircase=False,
      optimizer_adam_beta1=0.9,
      attention_dropout=0.0,
      min_length_bucket=8,
      summarize_grads=False,
      dropout=0.2,
      moe_hidden_sizes=2048,
      learning_rate_cosine_cycle_steps=250000,
      batch_size=6250,
      eval_drop_long_sequences=False,
      attention_value_channels=0,
      norm_epsilon=1e-06,
      nbr_decoder_problems=1,
      relu_dropout=0.0,
      scheduled_sampling_warmup_steps=50000,
      no_data_parallelism=False,
      norm_type="layer",
      moe_num_experts=64,
      relu_dropout_broadcast_dims="",
      target_modality="default",
      weight_decay=0.0,
      max_input_seq_length=0,
      optimizer_momentum_nesterov=False,
      factored_logits=False,
      min_length=0,
      optimizer_momentum_momentum=0.9,
      parameter_attention_value_channels=0,
      symbol_dropout=0.0,
      kernel_width=1,
      optimizer_adam_epsilon=1e-09,
      use_pad_remover=True,
      learning_rate_warmup_steps=2000,
      scheduled_sampling_gold_mixin_prob=0.5,
      problem_choice="adaptive",
      multiply_embedding_mode="sqrt_depth",
      num_heads=8,
      filter_size=2048,
      ffn_layer="dense_relu_dense",
      max_relative_position=0,
      attention_dropout_broadcast_dims="",
      length_bucket_step=1.1,
      learning_rate_boundaries=[0],
      shared_source_target_embedding=False
  )