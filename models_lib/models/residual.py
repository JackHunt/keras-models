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

from keras import Model, Sequential
from keras.layers import AveragePooling2D, Conv2D, Dense, MaxPool2D

from models_lib.layers.residual import ResidualBlock


class _ResNet(Model):
    def __init__(
        self,
        residual_blocks: typing.List[ResidualBlock],
        num_classes: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Initial "input" block.
        self._initial_block = Sequential(
            [
                Conv2D(
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    filters=64,
                    padding="same",
                    activation="relu",
                ),
                MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            ]
        )

        # Residual layers.
        for b in residual_blocks:
            if not isinstance(b, ResidualBlock):
                raise ValueError("Non ResidualBlock found.")

        self._residual_blocks = Sequential(residual_blocks)

        # Output classifier block.
        self._output_block = None
        if num_classes > 0:
            self._output_block = Sequential(
                [
                    AveragePooling2D(),
                    Dense(num_classes, activation="softmax"),
                ]
            )

    def call(self, inputs):
        y = self.initial_block(inputs)
        y = self.residual_blocks(y)

        if not self.output_block is None:
            return self.output_block(y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {"residual_blocks": self.residual_blocks, "num_classes": self.num_classes}
        )
        return config

    @property
    def initial_block(self) -> Sequential:
        return self._initial_block

    @property
    def residual_blocks(self) -> Sequential:
        return self._residual_blocks

    @property
    def output_block(self) -> Sequential:
        return self._output_block


class ResNet18(_ResNet):
    def __init__(self, **kwargs):
        blocks = []

        # Block 0.
        blocks += [ResidualBlock((3, 3), 64) for _ in range(2)]

        # Block 1.
        blocks += [
            ResidualBlock((3, 3), 128, shortcut_conv_depth=128),
            ResidualBlock((3, 3), 128),
        ]

        # Block 2.
        blocks += [
            ResidualBlock((3, 3), 256, shortcut_conv_depth=256),
            ResidualBlock((3, 3), 256),
        ]

        # Block 3.
        blocks += [
            ResidualBlock((3, 3), 512, shortcut_conv_depth=512),
            ResidualBlock((3, 3), 512),
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return ResNet18(**config)


class ResNet34(_ResNet):
    def __init__(self, **kwargs):
        blocks = []

        # Block 0.
        blocks += [ResidualBlock((3, 3), 64) for _ in range(3)]

        # Block 1.
        blocks.append(ResidualBlock((3, 3), 128, shortcut_conv_depth=128))
        blocks += [ResidualBlock((3, 3), 128) for _ in range(3)]

        # Block 2.
        blocks.append(ResidualBlock((3, 3), 256, shortcut_conv_depth=256))
        blocks += [ResidualBlock((3, 3), 256) for _ in range(5)]

        # Block 3.
        blocks.append(ResidualBlock((3, 3), 512, shortcut_conv_depth=512))
        blocks += [ResidualBlock((3, 3), 512) for _ in range(2)]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return ResNet34(**config)


class ResNet50(_ResNet):
    def __init__(self, **kwargs):
        blocks = []

        # Block 0.
        blocks += [ResidualBlock((3, 3), 64) for _ in range(2)]

        # Block 1.
        blocks += [
            ResidualBlock((3, 3), 128, shortcut_conv_depth=128),
            ResidualBlock((3, 3), 128),
        ]

        # Block 2.
        blocks += [
            ResidualBlock((3, 3), 256, shortcut_conv_depth=256),
            ResidualBlock((3, 3), 256),
        ]

        # Block 3.
        blocks += [
            ResidualBlock((3, 3), 512, shortcut_conv_depth=512),
            ResidualBlock((3, 3), 512),
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return ResNet50(**config)


class ResNet101(_ResNet):
    def __init__(self, **kwargs):
        blocks = []

        # Block 0.
        blocks.append(
            ResidualBlock((1, 1), 64, kernel_size_b=(3, 3), shortcut_conv_depth=64)
        )
        blocks += [ResidualBlock((1, 1), 64, kernel_size_b=(3, 3)) for _ in range(2)]

        # Block 1.
        blocks.append(
            ResidualBlock((1, 1), 128, kernel_size_b=(3, 3), shortcut_conv_depth=128)
        )
        blocks += [ResidualBlock((1, 1), 128, kernel_size_b=(3, 3)) for _ in range(3)]

        # Block 2.
        blocks.append(
            ResidualBlock((1, 1), 256, kernel_size_b=(3, 3), shortcut_conv_depth=256)
        )
        blocks += [ResidualBlock((1, 1), 256, kernel_size_b=(3, 3)) for _ in range(22)]

        # Block 3.
        blocks.append(
            ResidualBlock((1, 1), 512, kernel_size_b=(3, 3), shortcut_conv_depth=512)
        )
        blocks += [ResidualBlock((1, 1), 512, kernel_size_b=(3, 3)) for _ in range(2)]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return ResNet101(**config)


class ResNet152(_ResNet):
    def __init__(self, **kwargs):
        blocks = []

        # Block 0.
        blocks.append(
            ResidualBlock((1, 1), 64, kernel_size_b=(3, 3), shortcut_conv_depth=64)
        )
        blocks += [ResidualBlock((1, 1), 64, kernel_size_b=(3, 3)) for _ in range(2)]

        # Block 1.
        blocks.append(
            ResidualBlock((1, 1), 128, kernel_size_b=(3, 3), shortcut_conv_depth=128)
        )
        blocks += [ResidualBlock((1, 1), 128, kernel_size_b=(3, 3)) for _ in range(7)]

        # Block 2.
        blocks.append(
            ResidualBlock((1, 1), 256, kernel_size_b=(3, 3), shortcut_conv_depth=256)
        )
        blocks += [ResidualBlock((1, 1), 256, kernel_size_b=(3, 3)) for _ in range(35)]

        # Block 3.
        blocks.append(
            ResidualBlock((1, 1), 512, kernel_size_b=(3, 3), shortcut_conv_depth=512)
        )
        blocks += [ResidualBlock((1, 1), 512, kernel_size_b=(3, 3)) for _ in range(2)]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return ResNet152(**config)


def resnet(
    arch: int = 18, num_classes: int = 0
) -> typing.Union[ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]:
    if arch == 18:
        return ResNet18(num_classes=num_classes)

    if arch == 34:
        return ResNet34(num_classes=num_classes)

    if arch == 50:
        return ResNet50(num_classes=num_classes)

    if arch == 101:
        return ResNet101(num_classes=num_classes)

    if arch == 152:
        return ResNet152(num_classes=num_classes)

    raise ValueError("Invalid ResNet architecture.")
