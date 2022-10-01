# BSD 3-Clause License

# Copyright (c) 2022, Jack Hunt
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import keras
from models_lib.layers.utils import sequential

class MLP(keras.Model):
  def __init__(self, hidden_sizes, output_size, hidden_act='relu',
               output_act='relu', **kwargs):
    super().__init__(**kwargs)

    self._hidden_sizes = hidden_sizes
    self._output_size = output_size
    self._hidden_act = hidden_act
    self._output_act = output_act

    if self.hidden_sizes:
      if any(n <= 0 for n in self.hidden_sizes):
        raise ValueError(
          "All hidden layer sizes must be greater than or equal to one.")

      self._hidden = sequential.SequentialLayer(
        [keras.layers.Dense(
          n, activation=self.hidden_act) for n in self.hidden_sizes])
    else:
      self._hidden = None

    if self.output_size <= 0:
      raise ValueError(
          "Output layer size must be greater than or equal to one.")

    self._output_layer = keras.layers.Dense(
      self.output_size, activation=self.output_act)

  def call(self, inputs):
    y = self.hidden_layers(inputs)
    return self.output_layer(y)

  def get_config(self):
    config = super().get_config()
    config.update({
      'hidden_sizes': self.hidden_sizes,
      'output_size': self.output_size,
      'hidden_act': self.hidden_act,
      'output_act': self.output_act
    })
    return config

  @classmethod
  def from_config(cls, config):
    return MLP(**config)

  @property
  def hidden_sizes(self):
    return self._hidden_sizes

  @property
  def output_size(self):
    return self._output_size

  @property
  def hidden_act(self):
    self._hidden_act

  @property
  def output_act(self):
    return self._output_act

  @property
  def hidden_layers(self):
    return self._hidden

  @property
  def output_layer(self):
    return self._output_layer