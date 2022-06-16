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

import sys
sys.path.append('..')

import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

from functools import partial

from models_lib.models.vgg import vgg
from models_lib.models.residual import resnet

def create_dataset(batch_size=16, dtype=tf.float32):
  def data_pipeline(ds, training=True):
    f = lambda x,t: (tf.cast(x, dtype) / 255., t)

    ds = ds_train.map(f, num_parallel_calls=tf.data.AUTOTUNE)
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

  return data_pipeline(ds_train), data_pipeline(ds_test, False)


def create_model(model_type, arch):
  if model_type in ('resnet', 'Resnet', 'ResNet'):
    resnet_archs = (18, 34, 50, 101, 152)
    if not arch in resnet_archs:
      raise ValueError(
        "Invalid Resnet architecture. Architecture must be one of %s"
        % resnet_archs)

    return resnet(arch)
  
  if model_type in ('vgg', 'VGG'):
    vgg_archs = (11, 13, 16, 19)
    if not arch in vgg_archs:
      raise ValueError(
        "Invalid VGG architecture. Architecture must be one of %s"
        % vgg_archs)

    return vgg(arch)

  raise ValueError("Model type %s is invalid." % model_type)

def opt_fn(lr):
  return tf.keras.optimizers.SGD(lr)

def train_model(model_fn, data_fn, epochs, learning_rate):
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    ds_train, ds_test = data_fn()
    model = model_fn()

    model.compile(opt_fn(learning_rate), 'mse')
    model.fit(ds_train,
              epochs=epochs)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Train a net on MNIST.')
  parser.add_argument('model_type', type=str)
  parser.add_argument('model_arch', type=int)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=0.01)

  args = parser.parse_args()

  model_type = args.model_type
  model_arch = args.model_arch
  model_fn = partial(create_model, model_type, model_arch)

  batch_size = args.batch_size
  dataset_fn = partial(create_dataset, batch_size)

  num_epochs = args.epochs
  learning_rate = args.learning_rate

  train_model(model_fn,
              dataset_fn,
              epochs=num_epochs,
              learning_rate=learning_rate)
