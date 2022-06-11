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

class SequentialLayer(keras.layers.Layer):
  """This class provides a layer that serves to encapsulate a collection
  of layers to be executed in a sequential manner. This class provides
  a single-input, single-output "block" of constituent layers, for which
  the input to layer `n+1` is the output of layer `n`.

  Arguments:
    layers: A list of layers to be executed in sequence.
  """
  def __init__(self, layers):
    super().__init__()

    if not layers:
      raise ValueError("No layers provided.")

    self._layers = layers

  def call(self, inputs):
    if not (self._layers and self.built):
      raise RuntimeError("Layer has not been built.")

    x = inputs
    y = None
    for l in self._layers:
      y = l(x)
      x = y
    return y

  def get_config(self):
    config = super().get_config()
    config.update({
      'layers': self._layers
    })
    return config

  @classmethod
  def from_config(cls, config):
    return SequentialLayer(**config)

  @property
  def layers(self):
    return self._layers