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

"""Trains and tests a CNN model on the CIFAR100 dataset.

Raises:
    ValueError: If an invalid batch size is provided.
    ValueError: If an invalid learning rate is provided.
"""

import argparse
import typing
from functools import partial

import keras

import tensorflow as tf

from models_lib.models.googlenet import GoogLeNet
from models_lib.models.residual import resnet
from models_lib.models.vgg import vgg
from models_lib.utils import data


def create_model(m_type: str, m_arch: int, num_classes: int = 100) -> tf.keras.Model:
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
        if m_arch is None:
            m_arch = 18

        resnet_archs = (18, 34, 50, 101, 152)
        if not m_arch in resnet_archs:
            raise ValueError(
                f"Invalid Resnet architecture. Architecture must be one of {resnet_archs}")
        if num_classes > 0:
            return keras.Sequential([
                resnet(m_arch),
                keras.layers.Flatten(),
                keras.layers.Dense(num_classes)
            ])
        return resnet(m_arch)

    if m_type in ('vgg', 'VGG'):
        if m_arch is None:
            m_arch = 11

        vgg_archs = (11, 13, 16, 19)
        if not m_arch in vgg_archs:
            raise ValueError(
                f"Invalid VGG architecture. Architecture must be one of {vgg_archs}")

        return vgg(m_arch, num_classes=num_classes)

    if m_type in ('googlenet, GoogLeNet'):
        return GoogLeNet(num_classes=num_classes)

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
        model_fn (typing.Callable): A function returning a `keras.Model`.
        data_fn (typing.Callable): A function returning a `tf.Dataset`.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        ds_train, ds_test = data_fn()
        model = model_fn()

        if isinstance(model, GoogLeNet):
            ds_train = data.duplicate_targets(ds_train, 3)
            ds_test = data.duplicate_targets(ds_test, 3)

        model.compile(opt_fn(learning_rate), 'sparse_categorical_crossentropy')

        model.fit(ds_train,
                  epochs=epochs)
        model.eval(ds_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a net on CIFAR100.')
    parser.add_argument('model_type', type=str)
    parser.add_argument('--model_arch', type=int, default=None)
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

    dataset_fn = partial(data.create_cifar_100, bs, tf.float16, (48, 48))

    num_epochs = args.epochs

    lr = args.learning_rate
    if lr <= 0:
        raise ValueError("A nonzero, nonnegative learning rate is required.")

    train_model(model_func,
                dataset_fn,
                epochs=num_epochs,
                learning_rate=lr)
