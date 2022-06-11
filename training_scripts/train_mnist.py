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

from models_lib.models.vgg import vgg
from models_lib.models.residual import resnet

def create_dataset(batch_size=16):
  pass

def create_resnet(arch=18):
  pass

def train_model(model, epochs, learning_rate):
  pass

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Train a net on MNIST.')
  parser.add_argument('model_type', type=str)
  parser.add_argument('model_arch', type=int)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=0.01)

  args = parser.parse_args()

  model_type = args.model_type
  model_arch = args.model_arch
  if model_type == 'resnet':
    pass
  elif model_type == 'vgg':
    pass
  else:
    raise ValueError("Model type %s is invalid." % model_type)

  num_epochs = args.epochs
  learning_rate = args.learning_rate
