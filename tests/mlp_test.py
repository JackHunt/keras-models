# BSD 3-Clause License

# Copyright (c) 2021, Jack Hunt
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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from models_lib.models.mlp import MLP

import test_utils

class MLPTests(tf.test.TestCase):
  def test_bad_hidden_count(self):
    with self.assertRaisesRegex(
      ValueError,
      "All hidden layer sizes must be greater than or equal to one."):
      _ = MLP([1, 2, 0, 3], 3)

  def test_bad_output_count(self):
    with self.assertRaisesRegex(
      ValueError,
      "Output layer size must be greater than or equal to one."):
      _ = MLP([1, 2, 3], 0)
  
  def test_simple_train(self):
    x = np.ones((8, 32), dtype=np.float16)
    y = x * 2.0

    input_layer = keras.layers.Input(x.shape[1:], dtype=x.dtype)
    
    mlp = MLP([8, 16], 32)(input_layer)
    
    model = keras.Model(inputs=input_layer, outputs=mlp)
    model.compile('sgd', 'mse')

    history = model.fit(x, y, verbose=False, epochs=3)
    losses = history.history['loss']

    self.assertTrue(test_utils.monotonically_decreasing(losses))

if __name__ == '__main__':
  tf.test.main()