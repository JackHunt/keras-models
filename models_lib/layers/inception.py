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

import tensorflow as tf

from models_lib.layers.utils.sequential import SequentialLayer

class InceptionBlock(keras.layers.Layer):
    def __init__(self,
                 num_filters_1x1: int,
                 num_filters_3x3: int,
                 num_filters_5x5: int,
                 num_filters_3x3_dim_reduce: int = 0,
                 num_filters_5x5_dim_reduce: int = 0,
                 max_pool: bool = False,
                 num_filters_max_pool_dim_reduce: int = 0):
        super().__init__()

        self._n_1x1 = num_filters_1x1
        self._n_3x3 = num_filters_3x3
        self._n_5x5 = num_filters_5x5
        self._n_3x3_r = num_filters_3x3_dim_reduce
        self._n_5x5_r = num_filters_5x5_dim_reduce
        self._use_mp = max_pool
        self._n_mp_filters = num_filters_max_pool_dim_reduce

        # 1x1 conv.
        branch_0_layers = [
            keras.layers.Conv2D(
                self.num_filters_1x1, (1, 1), padding='same', activation='relu')
        ]
        self._branch_0 = SequentialLayer(branch_0_layers)

        # 3x3 conv.
        branch_1_layers = [
            keras.layers.Conv2D(
                self.num_filters_3x3, (3, 3), padding='same', activation='relu')
        ]
        if self.num_filters_3x3_dim_reduce:
            branch_1_layers.insert(0, keras.layers.Conv2D(
                self.num_filters_3x3_dim_reduce, (1, 1),
                padding='same', activation='relu'))
        self._branch_1 = SequentialLayer(branch_1_layers)

        # 5x5 conv.
        branch_2_layers = [
            keras.layers.Conv2D(
                self.num_filters_5x5, (5, 5), padding='same', activation='relu')
        ]
        if self.num_filters_5x5_dim_reduce:
            branch_2_layers.insert(0, keras.layers.Conv2D(
                self.num_filters_5x5_dim_reduce, (1, 1),
                padding='same', activation='relu'))
        self._branch_2 = SequentialLayer(branch_2_layers)

        # Max Pool & 1x1 branch.
        self._branch_3 = None
        if self.max_pool:
            branch_3_layers = [
                keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
            ]
            if self.num_filters_max_pool_dim_reduce:
                branch_3_layers.insert(0, keras.layers.Conv2D(
                    self.num_filters_max_pool_dim_reduce, (1, 1),
                    padding='same', activation='relu'))
            self._branch_3 = SequentialLayer(branch_3_layers)

    def call(self, inputs):
        b0_out = self._branch_0(inputs)
        b1_out = self._branch_1(inputs)
        b2_out = self._branch_2(inputs)

        out = tf.concat([b0_out, b1_out, b2_out], 3)

        if self.max_pool:
            b3_out = self._branch_3(inputs)
            out = tf.concat([out, b3_out], 3)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters_1x1': self.num_filters_1x1,
            'num_filters_3x3': self.num_filters_3x3,
            'num_filters_5x5': self.num_filters_5x5,
            'num_filters_3x3_dim_reduce': self.num_filters_3x3_dim_reduce,
            'num_filters_5x5_dim_reduce': self.num_filters_5x5_dim_reduce,
            'max_pool': self.max_pool,
            'num_filters_max_pool_dim_reduce': self.num_filters_max_pool_dim_reduce
        })
        return config

    @classmethod
    def from_config(cls, config):
        return InceptionBlock(**config)

    @property
    def num_filters_1x1(self) -> int:
        return self._n_1x1

    @property
    def num_filters_3x3(self) -> int:
        return self._n_3x3

    @property
    def num_filters_3x3_dim_reduce(self) -> int:
        return self._n_3x3_r

    @property
    def num_filters_5x5(self) -> int:
        return self._n_5x5

    @property
    def num_filters_5x5_dim_reduce(self) -> int:
        return self._n_5x5_r

    @property
    def max_pool(self) -> bool:
        return self._use_mp

    @property
    def num_filters_max_pool_dim_reduce(self) -> int:
        return self._n_mp_filters
