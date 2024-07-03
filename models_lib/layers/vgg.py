# BSD 3-Clause License

# Copyright (c) 2024, Jack Hunt
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

from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D


class VGGBlock(Sequential):
    def __init__(self, num_convolutions: int, num_filters: int):
        self._num_convolutions = num_convolutions
        if self.num_convolutions < 1:
            raise ValueError(
                "VGGBlock must have a positive, nonzero convolution count."
            )

        self._num_filters = num_filters
        if self.num_filters < 1:
            raise ValueError("VGGBlock convolutions must have at least one channel.")

        layers = []
        for _ in range(self.num_convolutions):
            layers.append(
                Conv2D(
                    filters=self.num_filters,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                )
            )

        layers.append(MaxPool2D(pool_size=2, strides=2))

        super().__init__(layers)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_convolutions": self._num_convolutions,
                "num_filters": self._num_filters,
            }
        )
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


class VGGClassifier(Sequential):
    def __init__(self, num_classes: int):
        if num_classes < 0:
            raise ValueError("VGGClassifier num_classes must be nonnegative.")

        self._num_classes = num_classes

        layers = [
            Flatten(),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(self.num_classes, activation="softmax"),
        ]

        super().__init__(layers)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return VGGClassifier(**config)

    @property
    def num_classes(self) -> int:
        return self._num_classes
