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

from tensorflow import keras
from models_lib.layers import residual
from models_lib.layers.utils import sequential

class _ResNet(keras.Model):
  def __init__(self, arch, **kwargs):
    super().__init__(**kwargs)

    self._arch = arch

    # Initial "input" block.
    self._initial_block = sequential.SequentialLayer([
      keras.layers.Conv2D(kernel_size=(7, 7),
                          strides=(2, 2),
                          filters=64,
                          padding='same',
                          activation='relu'),
      keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
    ])

    # Residual blocks.
    blk = []
    for c, n in self.arch:
      blk.append(sequential.SequentialLayer([residual.ResidualBlock(**c)] * n))
    self._residual_blocks = sequential.SequentialLayer(blk)

    # Output classifier block.
    self._output_block = sequential.SequentialLayer([
      keras.layers.AveragePooling2D(),
      keras.layers.Dense(1000, activation='relu'),
      keras.layers.Softmax()
    ])

  def call(self, inputs, training, mask):
    y = self.initial_block(inputs)
    y = self.residual_blocks(y)
    return self.output_block(y)

  def get_config(self):
    config = super().get_config()
    config.update({
      'arch': self.arch
    })
    return config

  @property
  def arch(self):
    return self._arch

  @property
  def initial_block(self):
    return self._initial_block

  @property
  def residual_blocks(self):
    return self._residual_blocks

  @property
  def output_block(self):
    return self._output_block

class ResNet18(_ResNet):
  def __init__(self, **kwargs):
    arch = [
      # Block 0.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 64
        },
        2
      ),
      # Block 1.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 128
        },
        2
      ),
      # Block 2.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 256
        },
        2
      ),
      # Block 3.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 512
        },
        2
      )
    ]
    
    super().__init__(arch, **kwargs)

  @classmethod
  def from_config(cls, config):
    return ResNet18(**config)

class ResNet34(_ResNet):
  def __init__(self, **kwargs):
    arch = [
      # Block 0.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 64
        },
        3
      ),
      # Block 1.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 128
        },
        4
      ),
      # Block 2.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 256
        },
        6
      ),
      # Block 3.
      (
        {
          'kernel_size': (3, 3),
          'num_filters': 512
        },
        3
      )
    ]

    super().__init__(arch, **kwargs)

  @classmethod
  def from_config(cls, config):
    return ResNet34(**config)

class ResNet50(_ResNet):
  def __init__(self, **kwargs):
    arch = [
      # Block 0.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 64,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 256
        },
        3
      ),
      # Block 1.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 128,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 512
        },
        4
      ),
      # Block 2.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 256,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 1024
        },
        6
      ),
      # Block 3.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 512,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 2048
        },
        3
      ),
    ]

    super().__init__(arch, **kwargs)

  @classmethod
  def from_config(cls, config):
    return ResNet50(**config)

class ResNet101(_ResNet):
  def __init__(self, **kwargs):
    arch = [
      # Block 0.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 64,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 256
        },
        3
      ),
      # Block 1.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 128,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 512
        },
        4
      ),
      # Block 2.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 256,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 1024
        },
        23
      ),
      # Block 3.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 512,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 2048
        },
        3
      ),
    ]
    
    super().__init__(arch, **kwargs)

  @classmethod
  def from_config(cls, config):
    return ResNet101(**config)

class ResNet152(_ResNet):
  def __init__(self, **kwargs):
    arch = [
      # Block 0.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 64,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 256
        },
        3
      ),
      # Block 1.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 128,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 512
        },
        8
      ),
      # Block 2.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 256,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 1024
        },
        36
      ),
      # Block 3.
      (
        {
          'kernel_size': (1, 1),
          'num_filters': 512,
          'kernel_size_b': (3, 3),
          'num_downsample_filters': 2048
        },
        3
      ),
    ]

    super().__init__(arch, **kwargs)

  @classmethod
  def from_config(cls, config):
    return ResNet152(**config)

def resnet(arch=18):
  if arch == 18:
    return ResNet18()

  if arch == 34:
    return ResNet34()

  if arch == 50:
    return ResNet50()

  if arch == 101:
    return ResNet101()

  if arch == 152:
    return ResNet152()

  raise ValueError("Invalid ResNet architecture.")