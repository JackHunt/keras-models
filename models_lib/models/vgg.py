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
from models_lib.layers import vgg
from models_lib.layers.utils import sequential

class _VGGNet(keras.Model):
  def __init__(self, arch, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._vgg_blocks = sequential.SequentialLayer(
      [vgg.VGGBlock(*nd) for nd in arch])

    self._classifier_block = sequential.SequentialLayer([
      keras.layers.Flatten(),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dense(self.num_classes, activation='softmax')
    ])

  def call(self, inputs, training, mask, features_only=False):
    features = self.vgg_blocks(inputs)
    if features_only:
      return features
    return self.classifier_block(features)

  @property
  def vgg_blocks(self):
    return self._vgg_blocks

  @property
  def classifier_block(self):
    return self._classifier_block

  @property
  def num_classes(self):
    return self._num_classes

class VGG11(_VGGNet):
  def __init__(self, *args, **kwargs):
    arch = [
      (1, 64),
      (1, 128),
      (2, 256),
      (2, 512),
      (2, 512)
    ]

    super().__init__(arch, *args, **kwargs)

class VGG13(_VGGNet):
  def __init__(self, *args, **kwargs):
    arch = [
      (2, 64),
      (2, 128),
      (2, 256),
      (2, 512),
      (2, 512)
    ]

    super().__init__(arch, *args, **kwargs)

class VGG16(_VGGNet):
  def __init__(self, *args, **kwargs):
    arch = [
      (2, 64),
      (2, 128),
      (3, 256),
      (3, 512),
      (3, 512)
    ]

    super().__init__(arch, *args, **kwargs)

class VGG19(_VGGNet):
  def __init__(self, *args, **kwargs):
    arch = [
      (2, 64),
      (2, 128),
      (4, 256),
      (4, 512),
      (4, 512)
    ]

    super().__init__(arch, *args, **kwargs)