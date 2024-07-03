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

from typing import List, Tuple, Union

import keras


class MLP(keras.Model):
    def __init__(
        self,
        hidden_sizes: Union[List[int], Tuple[int]],
        output_size: int,
        hidden_activation: str = "relu",  # TODO: Union with ops
        output_activation: str = "relu",  # TODO: ditto
        name="MLP",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation

        if self.hidden_sizes:
            if any(n <= 0 for n in self.hidden_sizes):
                raise ValueError(
                    "All hidden layer sizes must be greater than or equal to one."
                )

            self._hidden = keras.Sequential(
                [
                    keras.layers.Dense(n, activation=self.hidden_activation)
                    for n in self.hidden_sizes
                ]
            )
        else:
            self._hidden = None

        if self.output_size <= 0:
            raise ValueError("Output layer size must be greater than or equal to one.")

        self._output_layer = keras.layers.Dense(
            self.output_size, activation=self.output_activation
        )

    def call(self, inputs):
        y = self.hidden_layers(inputs)
        return self.output_layer(y)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_sizes": self.hidden_sizes,
                "output_size": self.output_size,
                "hidden_activation": self.hidden_activation,
                "output_activation": self.output_activation,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return MLP(**config)

    @property
    def hidden_sizes(self) -> Union[List[int], Tuple[int]]:
        return self._hidden_sizes

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def hidden_activation(self) -> str:
        self._hidden_activation

    @property
    def output_activation(self) -> str:
        return self._output_activation

    @property
    def hidden_layers(self) -> keras.Sequential:
        return self._hidden

    @property
    def output_layer(self) -> keras.layers.Dense:
        return self._output_layer
