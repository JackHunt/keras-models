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

import typing

from keras import Sequential
from keras.layers import Add, BatchNormalization, Conv2D, Layer, ReLU


class ResidualBlock(Layer):
    def __init__(
        self,
        kernel_size: typing.Union[int, typing.Tuple[int, int]],
        num_filters: int,
        shortcut_conv_depth: int = 0,
        kernel_size_b: typing.Union[int, typing.Tuple[int, int]] = None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size
        if self.kernel_size[0] < 1 or self.kernel_size[1] < 1:
            raise ValueError(
                "ResidualBlock kernel_size must be greater than or equal to one in each dimension."
            )

        self._kernel_size_b = kernel_size_b
        if self.kernel_size_b is not None and isinstance(self.kernel_size_b, int):
            self._kernel_size_b = (self.kernel_size_b, self.kernel_size_b)

        if self.kernel_size_b:
            if self.kernel_size_b[0] < 1 or self.kernel_size_b[1] < 1:
                raise ValueError(
                    "ResidualBlock kernel_size_b must be greater than or equal to one in each dimension."
                )
        else:
            self._kernel_size_b = self.kernel_size

        self._num_filters = num_filters
        if self._num_filters < 1:
            raise ValueError(
                "ResidualBlock num_filters must be greater than or equal to one."
            )

        self._shortcut_conv_depth = shortcut_conv_depth
        self._downsampling = self.shortcut_conv_depth > 0

        self._conv = Sequential(
            [
                Conv2D(
                    kernel_size=self.kernel_size,
                    strides=(2, 2) if self.downsampling else (1, 1),
                    filters=self.num_filters,
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                Conv2D(
                    kernel_size=self.kernel_size_b,
                    strides=(1, 1),
                    filters=self.num_filters,
                    padding="same",
                    activation="relu",
                ),
            ]
        )

        if self.downsampling:
            self._ds = Conv2D(
                kernel_size=1,
                strides=2,
                filters=self.shortcut_conv_depth,
                padding="same",
                activation="relu",
            )

        self._bn = BatchNormalization()

    def call(self, inputs):
        y = self._conv(inputs)

        residual = inputs
        if self.downsampling:
            residual = self._ds(residual)

        y = Add()([y, residual])
        y = ReLU()(y)
        return self._bn(y)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self._kernel_size,
                "num_filters": self._num_filters,
                "shortcut_conv_depth": self._shortcut_conv_depth,
                "kernel_size_b": self._kernel_size_b,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return ResidualBlock(**config)

    @property
    def kernel_size(self) -> typing.Union[int, typing.Tuple[int, int]]:
        return self._kernel_size

    @property
    def kernel_size_b(self) -> typing.Union[int, typing.Tuple[int, int]]:
        return self._kernel_size_b

    @property
    def num_filters(self) -> int:
        return self._num_filters

    @property
    def downsampling(self) -> bool:
        return self._downsampling

    @property
    def shortcut_conv_depth(self) -> int:
        return self._shortcut_conv_depth
