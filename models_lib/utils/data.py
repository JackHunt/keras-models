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

import typing

import tensorflow as tf
import tensorflow_datasets as tfds

def normalize_resize_images(ds: tf.data.Dataset,
                            dtype: tf.DType = tf.float32,
                            target_shape: typing.Tuple[int, int] = None) -> tf.data.Dataset:
    f = lambda x, t: (tf.cast(x, dtype) / 255., t)

    g = lambda x, t: (x, t)
    if not target_shape is None:
        g = lambda x, t: (tf.image.resize(
            x, target_shape, method=tf.image.ResizeMethod.BICUBIC), t)

    ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(g, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

def shuffle_batch_prefetch(ds: tf.data.Dataset,
                           batch_size: int = 32,
                           ds_info: tfds.core.DatasetInfo = None,
                           training: bool = True) -> tf.data.Dataset:
    ds = ds.cache()

    if training and not ds_info is None:
        ds = ds.shuffle(ds_info.splits['train'].num_examples)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def duplicate_targets(ds: tf.data.Dataset, n: int) -> tf.data.Dataset:
    return ds.map(lambda x, t: (x, tuple(t for _ in range(n))),
                  num_parallel_calls=tf.data.AUTOTUNE)

def create_cifar_100(batch_size: int = 16,
                     dtype: tf.DType = tf.float32,
                     target_shape: typing.Tuple[int, int] = None) -> tf.data.Dataset:
    """Creates two CIFAR100 datasets, one for training and one
    for testing.

    Args:
        batch_size (int, optional): The batch size for the dataset. Defaults to 16.
        dtype (tf.DType, optional): The dtype of the output data. Defaults to tf.float32.
        target_shape (tuple, optional): Shape to which the images should be reshaped.
    """
    target_shape = tf.convert_to_tensor(target_shape) if target_shape else None

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar100',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    ds_train = normalize_resize_images(ds_train,
                                       dtype=dtype,
                                       target_shape=target_shape)
    ds_train = shuffle_batch_prefetch(ds_train,
                                      batch_size=batch_size,
                                      ds_info=ds_info)

    ds_test = normalize_resize_images(ds_test,
                                      dtype=dtype,
                                      target_shape=target_shape)
    ds_test = shuffle_batch_prefetch(ds_test,
                                     batch_size=batch_size,
                                     training=False)

    return ds_train, ds_test