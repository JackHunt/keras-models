# BSD 3-Clause License

# Copyright (c) 2021, Jack Hunt
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

from tensorflow import keras
from tensorflow.keras import layers
class VGGLayer(keras.layers.Layer):
  def __init__(self, num_convolutions, num_channels):
    super().__init__()

    self._num_convolutions = num_convolutions
    if self.num_convolutions <= 0:
      raise ValueError(
        "VGGLayer must have a positive, nonzero convolution count.")

    self._num_channels = num_channels
    if self.num_channels < 1:
      raise ValueError(
        "VGGLayer convolutions must have at least one channel.")

    self._layers = []
    self._seq = None

  def call(self, inputs, *args, **kwargs):
      if not self._layers:
        raise RuntimeError("VGGLayer has not been built.")

      x = inputs
      y = None
      for l in self._layers:
        y = l(x)
        x = y
      return y

  def build(self, input_shape):
      super().build(input_shape)

      if self._layers:
        raise RuntimeError("VGGLayer has already been built.")

      for _ in range(self.num_convolutions):
        self._layers += [
          keras.layers.Conv2D(self.num_channels,
                              kernel_size=3,
                              padding='same',
                              activation='relu'),
          keras.layers.MaxPool2D(pool_size=2,
                                 strides=2)
        ]

  @property
  def num_convolutions(self):
    return self._num_convolutions

  @property
  def num_channels(self):
    return self._num_channels