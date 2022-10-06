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
from models_lib.layers.inception import InceptionBlock
from models_lib.layers.utils.sequential import SequentialLayer

class GoogLeNet(keras.Model):
    """This class implements the GoogLeNet model.
    """
    def __init__(self):
        super().__init__()

        # Block 0.
        self._block_0 = SequentialLayer([
            keras.layers.Conv2D(64, (7, 7),
                                padding='same',
                                strides=(2, 2),
                                activation='relu'),
            keras.layers.MaxPool2D((3, 3),
                                   padding='same',
                                   strides=(2, 2)),
            keras.layers.Conv2D(64, (1, 1),
                                padding='same',
                                strides=(1, 1),
                                activation='relu'),
            keras.layers.Conv2D(192, (3, 3),
                                padding='same',
                                strides=(1, 1),
                                activation='relu'),
            keras.layers.MaxPool2D((3, 3),
                                   padding='same',
                                   strides=(2, 2)),
            InceptionBlock(64, 128, 32,
                           num_filters_3x3_dim_reduce=96,
                           num_filters_5x5_dim_reduce=16,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=32),
            InceptionBlock(128, 192, 96,
                           num_filters_3x3_dim_reduce=128,
                           num_filters_5x5_dim_reduce=32,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=64),
            keras.layers.MaxPool2D((3, 3),
                                   padding='same',
                                   strides=(2, 2)),
            InceptionBlock(192, 208, 48,
                           num_filters_3x3_dim_reduce=96,
                           num_filters_5x5_dim_reduce=16,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=64)
        ])

        # Output Block 0.
        self._output_block_0 = SequentialLayer([
            keras.layers.AveragePooling2D((5, 5),strides=3),
            keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Block 1.
        self._block_1 = SequentialLayer([
            InceptionBlock(160, 224, 64,
                           num_filters_3x3_dim_reduce=112,
                           num_filters_5x5_dim_reduce=24,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=64),
            InceptionBlock(128, 256, 64,
                           num_filters_3x3_dim_reduce=128,
                           num_filters_5x5_dim_reduce=24,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=64),
            InceptionBlock(112, 288, 64,
                           num_filters_3x3_dim_reduce=144,
                           num_filters_5x5_dim_reduce=32,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=64)
        ])

        # Output Block 1.
        self._output_block_1 = SequentialLayer([
            keras.layers.AveragePooling2D((5, 5), strides=3),
            keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Block 2.
        self._block_2 = SequentialLayer([
            InceptionBlock(256, 320, 128,
                           num_filters_3x3_dim_reduce=160,
                           num_filters_5x5_dim_reduce=32,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=128),
            keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2)),
            InceptionBlock(384, 384, 128,
                           num_filters_3x3_dim_reduce=192,
                           num_filters_5x5_dim_reduce=48,
                           max_pool=True,
                           num_filters_max_pool_dim_reduce=128),
        ])

        # Output Block 2.
        self._output_block_2 = SequentialLayer([
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        x = self._block_0(inputs)
        output_0 = self._output_block_0(x)

        x = self._block_1(x)
        output_1 = self._output_block_1(x)

        x = self._block_2(x)
        output_2 = self._output_block_2(x)

        return output_0, output_1, output_2

    def get_config(self):
        config = super().get_config()
        config.update({
            #
        })
        return config

    @classmethod
    def from_config(cls, config):
        return GoogLeNet(**config)
