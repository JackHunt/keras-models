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

from typing import Tuple

import tensorflow as tf


def _split_image(image: tf.Tensor, grid_shape: Tuple[int, int]) -> tf.Tensor:
    _, H, W, C = image.shape
    H_grid, W_grid = grid_shape
    if H % H_grid != 0 or W % W_grid != 0:
        raise ValueError("Image dimensions must be divisible by grid_shape")

    H_patch = H // H_grid
    W_patch = W // W_grid

    patches = tf.nn.space_to_depth(image, block_size=grid_shape)
    return tf.reshape(
        image,
        [
            H_grid * W_grid,
            H_patch,
            W_patch,
            C,
        ],
    )


def _reconstruct_image(patches: tf.Tensor, grid_shape: Tuple[int, int]) -> tf.Tensor:
    _, H_patch, W_patch, C = patches.shape
    H_grid, W_grid = grid_shape

    patches = tf.reshape(patches, [H_grid, W_grid, H_patch, W_patch, C])
    patches = tf.transpose(patches, [0, 2, 1, 3, 4])
    return tf.reshape(patches, [H_grid * H_patch, W_grid * W_patch, C])


def shuffle_jigsaw(
    image: tf.Tensor, grid_shape: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    patches = _split_image(image, grid_shape)

    H_grid, W_grid = grid_shape
    idx = tf.range(H_grid * W_grid)
    idx_shuffled = tf.random.shuffle(idx)

    patches_shuffled = tf.gather(patches, idx_shuffled)
    image_shuffled = _reconstruct_image(patches_shuffled, grid_shape)

    return image_shuffled, idx_shuffled


def unshuffle_jigsaw(
    image_shuffled: tf.Tensor,
    grid_shape: Tuple[int, int],
    idx_shuffled: tf.Tensor,
) -> tf.Tensor:
    patches_shuffled = _split_image(image_shuffled, grid_shape)

    idx = tf.argsort(idx_shuffled)
    patches = tf.gather(patches_shuffled, idx)

    return _reconstruct_image(patches, grid_shape)


class JigsawDataset:
    def __init__(
        self,
        ds: tf.data.Dataset,
        grid_shape: Tuple[int],
    ):
        self._ds = ds
        self._grid_shape = grid_shape

    def _preprocess(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return shuffle_jigsaw(image, self._grid_shape)

    def get_dataset(self) -> tf.data.Dataset:
        return self.ds.map(
            self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
