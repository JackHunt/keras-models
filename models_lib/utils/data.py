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
import tensorflow_datasets as tfds


def normalise_images(
    ds: tf.data.Dataset, dtype: tf.DType = tf.float32
) -> tf.data.Dataset:
    return ds.map(
        lambda x, t: (tf.cast(x, dtype) / 255.0, t), num_parallel_calls=tf.data.AUTOTUNE
    )


def resize_images(
    ds: tf.data.Dataset, target_shape: Tuple[int, int]
) -> tf.data.Dataset:
    return ds.map(
        lambda x, t: (
            tf.image.resize(x, target_shape, method=tf.image.ResizeMethod.BICUBIC),
            t,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def minmax_norm(ds: tf.data.Dataset, axis: int = 0) -> tf.data.Dataset:
    def minmax(x):
        x_min = tf.reduce_min(x, axis=axis)
        x_max = tf.reduce_max(x, axis=axis)

        return (x - x_min) / (x_max - x_min)

    return ds.map(lambda x, t: (minmax(x), t), num_parallel_calls=tf.data.AUTOTUNE)


def shuffle_batch_prefetch(
    ds: tf.data.Dataset,
    batch_size: int = 32,
    ds_info: tfds.core.DatasetInfo = None,
    training: bool = True,
) -> tf.data.Dataset:
    ds = ds.cache()

    if training and not ds_info is None:
        ds = ds.shuffle(
            ds_info.splits["train"].num_examples, reshuffle_each_iteration=False
        )

    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def duplicate_targets(ds: tf.data.Dataset, n: int) -> tf.data.Dataset:
    return ds.map(
        lambda x, t: (x, tuple(t for _ in range(n))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def make_categorical(ds: tf.data.Dataset, num_classes: int) -> tf.data.Dataset:
    return ds.map(
        lambda x, t: (x, tf.one_hot(t, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def load_tfds(
    name: str, train_only: bool = False
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], tfds.core.DatasetInfo]:
    return tfds.load(
        name,
        split=["train", "test"] if not train_only else ["train"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )


def split_dataset(
    ds: tf.data.Dataset, val_split: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    if val_split < 0 or val_split > 1:
        raise ValueError("val_split must be between 0 and 1")

    n = int(round(val_split * 10))

    def is_val(i, d):
        return i % n == 0

    def is_train(i, d):
        return not is_val(i, d)

    val_ds = ds.enumerate().filter(is_val).map(lambda _, d: d)
    train_ds = ds.enumerate().filter(is_train).map(lambda _, d: d)

    return train_ds, val_ds


def create_cifar_100(
    batch_size: int = 16,
    dtype: tf.DType = tf.float32,
    normalise: bool = True,
    target_shape: Tuple[int, int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    target_shape = tf.convert_to_tensor(target_shape) if target_shape else None

    (ds_train, ds_test), ds_info = load_tfds("cifar100")

    def preprocess(ds, training):
        ds = resize_images(
            normalise_images(ds) if normalise else ds, target_shape=target_shape
        )
        ds = make_categorical(ds, 100)
        return shuffle_batch_prefetch(
            ds,
            batch_size=batch_size,
            ds_info=ds_info if training else None,
            training=training,
        )

    return preprocess(ds_train, True), preprocess(ds_test, False)


def create_iris(
    batch_size: int = 16, normalise: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds, ds_info = load_tfds("iris", train_only=True)
    ds = make_categorical(ds[0], 3)
    ds = minmax_norm(ds) if normalise else ds
    return shuffle_batch_prefetch(ds, batch_size=batch_size)
