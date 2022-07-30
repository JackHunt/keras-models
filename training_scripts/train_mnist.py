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

"""Trains and tests a CNN model on the MNIST digits dataset.

Raises:
    ValueError: If an invalid batch size is provided.
    ValueError: If an invalid learning rate is provided.
"""

import argparse
import typing
from functools import partial

import tensorflow_datasets as tfds
import tensorflow as tf

from models_lib.models.residual import resnet
from models_lib.models.vgg import vgg


def create_dataset(batch_size: int = 16,
                   dtype: tf.DType = tf.float32,
                   target_shape: typing.Tuple[int, int] = None):
    """Creates two MNIST datasets, one for training and one
    for testing.

    Args:
        batch_size (int, optional): The batch size for the dataset. Defaults to 16.
        dtype (tf.DType, optional): The dtype of the output data. Defaults to tf.float32.
        target_shape (tuple, optional): Shape to which the images should be reshaped.
    """
    target_shape = tf.convert_to_tensor(target_shape) if target_shape else None

    def data_pipeline(ds: tf.data.Dataset,
                      training: bool = True):
        f = lambda x, t: (tf.cast(x, dtype) / 255., t)

        g = lambda x, t: (x, t)
        if not target_shape is None:
            g = lambda x, t: (tf.image.resize(
                x, target_shape, method=tf.image.ResizeMethod.BICUBIC), t)

        ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(g, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()

        if training:
            ds = ds.shuffle(ds_info.splits['train'].num_examples)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    return data_pipeline(ds_train), data_pipeline(ds_test, training=False)


def create_model(m_type: str, m_arch: int) -> tf.keras.Model:
    """_summary_

    Args:
        m_type (str): The string description of the model.
        m_arch (int): The integer identifier of the model architecture.

    Raises:
        ValueError: If an invalid model type is requested.
        ValueError: If an invalid model architecture is requested.

    Returns:
        tf.keras.Model: An initialised Keras model of the requested architecture.
    """
    if m_type in ('resnet', 'Resnet', 'ResNet'):
        resnet_archs = (18, 34, 50, 101, 152)
        if not m_arch in resnet_archs:
            raise ValueError(
                f"Invalid Resnet architecture. Architecture must be one of {resnet_archs}")

        return resnet(m_arch)

    if m_type in ('vgg', 'VGG'):
        vgg_archs = (11, 13, 16, 19)
        if not m_arch in vgg_archs:
            raise ValueError(
                f"Invalid VGG architecture. Architecture must be one of {vgg_archs}")

        return vgg(m_arch)

    raise ValueError(f"Model type {m_type} is invalid.")


def opt_fn(learning_rate: float):
    """Generates a Stochastic Gradient Descent optimiser.

    Args:
        learning_rate (float): The learning rate for gradient updates.

    Returns:
        tf.keras.optimizers.SGD: A Keras SGD optimiser.
    """
    return tf.keras.optimizers.Adam(learning_rate)


def train_model(model_fn: typing.Callable,
                data_fn: typing.Callable,
                epochs: int,
                learning_rate: float):
    """_summary_

    Args:
        model_fn (typing.Callable): _description_
        data_fn (typing.Callable): _description_
        epochs (int): _description_
        learning_rate (float): _description_
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        ds_train, ds_test = data_fn()
        model = model_fn()

        model.compile(opt_fn(learning_rate), 'mse')
        model.fit(ds_train,
                  epochs=epochs)

        model.eval(ds_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a net on MNIST.')
    parser.add_argument('model_type', type=str)
    parser.add_argument('model_arch', type=int)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    args = parser.parse_args()

    model_type = args.model_type
    model_arch = args.model_arch
    model_func = partial(create_model, model_type, model_arch)

    bs = args.batch_size
    if bs <= 0:
        raise ValueError("A nonzero, nonnegative batch size is required.")

    dataset_fn = partial(create_dataset, bs, tf.float16, (48, 48))

    num_epochs = args.epochs

    lr = args.learning_rate
    if lr <= 0:
        raise ValueError("A nonzero, nonnegative leanring rate is required.")

    train_model(model_func,
                dataset_fn,
                epochs=num_epochs,
                learning_rate=lr)
