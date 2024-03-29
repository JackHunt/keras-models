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
from models_lib.layers.utils.sequential import SequentialLayer

class VGGBlock(SequentialLayer):
    """This class implements the "VGG Block" component of the VGG-x
    family of CNN architectures. Each `VGGBlock` consists of the
    following components (executed in the given order).

    - A number of 2D Convolutions
    - 2D Max Pooling
    """
    def __init__(self,
                 num_convolutions: int,
                 num_filters: int):
        """Instantiates a new `VGGBlock`.

        Args:
            num_convolutions (int): The number of convolution kernels.
            num_filters (int): The number of filters to use in the convolutions.

        Raises:
            ValueError: If `num_convolutions` or `num_filters` is less than 1.
        """
        self._num_convolutions = num_convolutions
        if self.num_convolutions < 1:
            raise ValueError(
                "VGGBlock must have a positive, nonzero convolution count.")

        self._num_filters = num_filters
        if self.num_filters < 1:
            raise ValueError(
                "VGGBlock convolutions must have at least one channel.")

        layers = []
        for _ in range(self.num_convolutions):
            layers.append(keras.layers.Conv2D(filters=self.num_filters,
                                              kernel_size=3,
                                              padding='same',
                                              activation='relu'))
    
        layers.append(keras.layers.MaxPool2D(pool_size=2, strides=2))

        super().__init__(layers)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_convolutions': self._num_convolutions,
            'num_filters': self._num_filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return VGGBlock(**config)

    @property
    def num_convolutions(self) -> int:
        return self._num_convolutions

    @property
    def num_filters(self) -> int:
        return self._num_filters

class VGGClassifier(SequentialLayer):
    """This class implements the "VGG Classifier" component of the VGG-x
    family of CNN architectures. Each `VGGClassifier` consists of the 
    following components (executed in the given order).

    - A Flatten Layer
    - A Dense layer of 4096 units and relu activation
    - A Dense layer of 4096 units and relu activation
    - A Dense layer of `num_classes` units and softmax activation.

    Arguments:
        num_classes: The number of output classes.
    """
    def __init__(self, num_classes: int):
        """Instantiates a new `VGGClassifier`.

        Args:
            num_classes (int): The number of target classes.

        Raises:
            ValueError: If `num_classes` is less than zero.
        """
        if num_classes < 0:
            raise ValueError("VGGClassifier num_classes must be nonnegative.")

        self._num_classes = num_classes

        layers = [
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ]

        super().__init__(layers)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return VGGClassifier(**config)

    @property
    def num_classes(self) -> int:
        return self._num_classes
