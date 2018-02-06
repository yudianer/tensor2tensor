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
      noeval_use_test_set='',
        proximity_bias=False,
        dbgprofile=False,
        eval_early_stopping_steps=None,
        cloud_tpu_name='jack-tpu',
        parameter_attention_value_channels=0,
        sync=False,
        attention_value_channels=0,
        tmp_dir='/tmp/t2t_datagen',
        learning_rate_decay_steps=5000,
        enable_graph_rewriter=False,
        eval_run_autoregressive=False,
        noprofile='',
        symbol_modality_skip_top=False,
        noregistry_help='',
        nosync='',
        moe_loss_coef=0.01,
        nohelpfull='',
        layer_prepostprocess_dropout=0.1,
        prepend_mode='none',
        target_modality='default',
        worker_gpu_memory_fraction=0.95,
        schedule='continuous_train_and_eval',
        eval_drop_long_sequences=False,
        pos='timing',
        nodbgprofile='',
        nohelpshort='',
        relu_dropout_broadcast_dims='',
        scheduled_sampling_gold_mixin_prob=0.5,
        learning_rate_decay_staircase=False,
        layer_preprocess_sequence='n',
        multiply_embedding_mode='sqrt_depth',
        kernel_height=3,
        num_encoder_layers=6,
        max_input_seq_length=0,
        label_smoothing=0.1,
        shared_embedding_and_softmax_weights=False,
        nolocally_shard_to_cpu='',
        scheduled_sampling_prob=0.0,
        norm_type='layer',
        ps_replicas=0,
        layer_prepostprocess_dropout_broadcast_dims='',
        noexport_saved_model='',
        force_full_predict=False,
        eval_early_stopping_metric='loss',
        weight_decay=0.0,
        symbol_modality_num_shards=16,
        optimizer_adam_epsilon=1e-09,
        ffn_layer='dense_relu_dense',
        worker_id=0,
        tfdbg=False,
        hidden_size=512,
        nolog_device_placement='',
        max_target_seq_length=0,
        parsing_path='',
        dropout=0.2,
        data_dir='/media/device1.8t/jack/corpus/mn-zh/trans/mn-zh-split/t2t',
        registry_help=False,
        ps_job='/job:ps',
        cloud_vm_name='jack-vm',
        moe_num_experts=64,
        attention_dropout=0.0,
        summarize_vars=False,
        hparams_set='transformer_base_single_gpu',
        keep_checkpoint_every_n_hours=10000,
        summarize_grads=False,
        parameter_attention_key_channels=0,
        iterations_per_loop=100,
        learning_rate_minimum=None,
        optimizer_momentum_nesterov=False,
        noeval_run_autoregressive='',
        compress_steps=0,
        initializer='uniform_unit_scaling',
        use_fixed_batch_size=False,
        learning_rate_cosine_cycle_steps=250000,
        train_steps=200000,
        min_length=0,
        daisy_chain_variables=True,
        num_hidden_layers=6,
        max_length=256,
        nogenerate_data='',
        scheduled_sampling_warmup_steps=50000,
        clip_grad_norm=0.0,
        filter_size=2048,
        eval_early_stopping_metric_delta=0.1,
        learning_rate_decay_rate=1.0,
        nohelp='',
        nbr_decoder_problems=1,
        weight_noise=0.0,
        layer_postprocess_sequence='da',
        max_relative_position=0,
        nocloud_delete_on_done='',
        attention_key_channels=0,
        relu_dropout=0.0,
        worker_job='/job:localhost',
        problem_choice='adaptive',
        timit_paths='',
        save_checkpoints_secs=0,
        input_modalities='default',
        symbol_dropout=0.0,
        worker_replicas=1,
        use_pad_remover=True,
        eval_early_stopping_metric_minimize=True,
        model='transformer',
        sampling_temp=1.0,
        factored_logits=False,
        decode_hparams='',
        hparams='',
        worker_gpu=2,
        keep_checkpoint_max=5,
        batch_size=6250,
        learning_rate=1.0,
        nocloud_tpu='',
        eval_use_test_set=False,
        attention_dropout_broadcast_dims='',
        eval_steps=2000,
        optimizer='Adam',
        local_eval_frequency=1000,
        num_heads=8,
        hparams_range=None,
        grad_noise_scale=0.0,
        locally_shard_to_cpu=False,
        num_decoder_layers=6,
        nouse_tpu='',
        length_bucket_step=1.1,
        min_length_bucket=8,
        optimizer_momentum_momentum=0.9,
        optimizer_adam_beta1=0.9,
        norm_epsilon=1e-06,
        notfdbg='',
        export_saved_model=False,
        initializer_gain=1.0,
        kernel_width=1,
        split_to_length=0,
        tpu_num_shards=8,
        learning_rate_warmup_steps=16000,
        problems='translate_mnzh_by_32kbpe',
        moe_k=2,
        self_attention_type='dot_product',
        log_device_placement=False,
        output_dir='/media/device1.8t/jack/corpus/mn-zh/trans/mn-zh-split/t2t/transformer_base_single_gpu',
        noenable_graph_rewriter='',
        learning_rate_decay_scheme='noam',
        sampling_method='argmax',
        no_data_parallelism=False,
        master='',
        moe_hidden_sizes=2048,
        random_seed=1234,
        gpu_order='',
        ps_gpu=0,
        optimizer_adam_beta2=0.997
  )