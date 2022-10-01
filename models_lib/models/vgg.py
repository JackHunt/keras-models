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
from models_lib.layers.vgg import VGGBlock, VGGClassifier
from models_lib.layers.utils.sequential import SequentialLayer

class _VGGNet(keras.Model):
    def __init__(self, vgg_blocks, num_classes=0, **kwargs):
        super().__init__(**kwargs)

        for b in vgg_blocks:
            if not isinstance(b, VGGBlock):
                raise ValueError("Non VGGBlock found.")

        self._vgg_blocks = SequentialLayer(vgg_blocks)

        self._classifier_block = None
        if num_classes > 0:
            self._classifier_block = VGGClassifier(num_classes)

    def call(self, inputs):
        features = self.vgg_blocks(inputs)
        if not self.classifier_block:
            return features
        return self.classifier_block(features)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vgg_blocks': self.vgg_blocks,
            'num_classes': self.classifier_block.num_classes
        })
        return config

    @property
    def vgg_blocks(self):
        return self._vgg_blocks

    @property
    def classifier_block(self):
        return self._classifier_block

class VGG11(_VGGNet):
    def __init__(self, **kwargs):
        blocks = [
            VGGBlock(1, 64),
            VGGBlock(1, 128),
            VGGBlock(2, 256),
            VGGBlock(2, 512),
            VGGBlock(2, 512)
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return VGG11(**config)

class VGG13(_VGGNet):
    def __init__(self, **kwargs):
        blocks = [
            VGGBlock(2, 64),
            VGGBlock(2, 128),
            VGGBlock(2, 256),
            VGGBlock(2, 512),
            VGGBlock(2, 512)
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return VGG13(**config)

class VGG16(_VGGNet):
    def __init__(self, **kwargs):
        blocks = [
            VGGBlock(2, 64),
            VGGBlock(2, 128),
            VGGBlock(3, 256),
            VGGBlock(3, 512),
            VGGBlock(3, 512)
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return VGG16(**config)

class VGG19(_VGGNet):
    def __init__(self, **kwargs):
        blocks = [
            VGGBlock(2, 64),
            VGGBlock(2, 128),
            VGGBlock(4, 256),
            VGGBlock(4, 512),
            VGGBlock(4, 512)
        ]

        super().__init__(blocks, **kwargs)

    @classmethod
    def from_config(cls, config):
        return VGG19(**config)

def vgg(arch=11, num_classes=0):
    if arch == 11:
        return VGG11(num_classes=num_classes)
  
    if arch == 13:
        return VGG13(num_classes=num_classes)

    if arch == 16:
        return VGG16(num_classes=num_classes)

    if arch == 19:
        return VGG19(num_classes=num_classes)

    raise ValueError("Invalid VGG architecture.")
