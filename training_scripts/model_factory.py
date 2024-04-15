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

from typing import Dict, Union

import keras

from models_lib.models.googlenet import GoogLeNet
from models_lib.models.mlp import MLP
from models_lib.models.residual import resnet, _ResNet
from models_lib.models.vgg import vgg, _VGGNet

def create_mlp(spec: Dict) -> MLP:
  return MLP(**spec)

def create_resnet(spec: Dict) -> Union[_ResNet, keras.Sequential]:
  resnet_size = spec['size']
  if not resnet_size in (18, 34, 50, 101, 152): # TODO: Does this need moving?
    raise ValueError(f"Invalid Resnet architecture: {resnet_size}.")

  num_classes = spec['num_classes']
  if not num_classes:
    return resnet(resnet_size)

  assert num_classes > 0

  return keras.Sequential([
    resnet(resnet_size),
    keras.layers.Flatten(),
    keras.layers.Dense(num_classes) # TODO: Replace with MLP class & softmax
  ])

def create_vgg(spec: Dict) -> _VGGNet:
  vgg_arch = spec['arch']
  if not vgg_arch in (11, 13, 16, 19):
    raise ValueError(f"Invalid VGG architecture: {vgg_arch}.")

  return vgg(**spec)

def create_googlenet() -> GoogLeNet:
  return GoogLeNet(**spec)

def create_model(architecture: Dict) -> keras.Model:
  arch_type = architecture['arch_type'].lower()
  spec = architecture['spec']

  if arch_type == "mlp":
    return create_mlp(spec)

  if arch_type == "resnet":
    return create_resnet(spec)

  if arch_type == "vgg":
    return create_vgg(spec)

  if arch_type == "googlenet":
    return create_googlenet(**spec)

  raise ValueError(f"Model type {arch_type} is invalid.")
