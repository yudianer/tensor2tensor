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


@registry.register_hparams("jack_basic_params")
def jack_basic_params():
  """
  A set of basic hyperparameters from jack_basic_params.json set by Jack
  """
  hparams = tf.contrib.training.HParams()
  jack_basic_params = json.load(open('jack_t2t_params.json'))
  for key in jack_basic_params.keys():
      value = jack_basic_params[key]
      if value:
        hparams.add_hparam(key, value)
  return jack_basic_params