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
from models_lib.layers.utils import sequential

class ResidualBlock(keras.layers.Layer):
  """This class implements the "Residual Block" component of the ResNet
  family of CNN architectures. Each `ResidualBlock` consists of the 
  following components (executed in the given order).

  - 2D Convolution
  - Batch Normalization
  - 2D Convolution (stride 1)
  - Optional 2D Convolution (kernel size 1, stride 2, for downsampling)
  - Batch Normalization

  Arguments:
    kernel_size: The size of the 2D convolution kernels.
    num_filters: The number of filters to use in the first two convolutions.
    num_downsample_filters: The number of filters to use in the third,
      downsampling convolution. Defaults to 0, disabling downsampling.
    kernel_size_b: An optional kernel size for the second convolution, should
      it need to differ from the first.
  """
  def __init__(self, kernel_size, num_filters,
               num_downsample_filters=0, kernel_size_b=None):
    super().__init__()

    self._kernel_size = kernel_size
    if self.kernel_size < 1:
      raise ValueError(
        "ResidualBlock kernel_size must be greater than or equal to one.")

    self._kernel_size_b = kernel_size_b
    if self.kernel_size_b:
      if self._kernel_size_b < 1:
        raise ValueError(
          "ResidualBlock kernel_size_b must be greater than or equal to one.")
    else:
      self._kernel_size_b = self.kernel_size

    self._num_filters = num_filters
    if self._num_filters < 1:
      raise ValueError(
        "ResidualBlock num_filters must be greater than or equal to one.")

    self._num_downsample_filters = num_downsample_filters
    self._downsampling = self.num_downsample_filters > 0

    self._conv = sequential.SequentialLayer(
      [
        keras.layers.Conv2D(kernel_size=self.kernel_size,
                            strides=(2, 2) if self.downsampling else (1, 1),
                            filters=self.num_filters,
                            padding='same',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(kernel_size=self.kernel_size_b,
                            strides=(1, 1),
                            filters=self.num_filters,
                            padding='same',
                            activation='relu')
      ])

    if self.downsampling:
      self._ds = keras.layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.num_downsample_filters,
                                     padding='same',
                                     activation='relu')

    self._bn = keras.layers.BatchNormalization()

  def call(self, inputs):
    y = self._conv(inputs)
    
    if self.downsampling:
      inputs = self._ds(inputs)
    
    y = keras.layers.Add()([y, inputs])
    y = keras.layers.ReLU()(y)
    return self._bn(y)

  @property
  def kernel_size(self):
    return self._kernel_size

  @property
  def kernel_size_b(self):
    return self._kernel_size_b

  @property
  def num_filters(self):
    return self._num_filters

  @property
  def downsampling(self):
    return self._downsampling

  @property
  def num_downsample_filters(self):
    return self._num_downsample_filters