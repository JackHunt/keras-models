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

import argparse

from pathlib import Path
from typing import Dict, Tuple

import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import yaml

from dataset_factory import create_dataset
from model_factory import create_model

def get_optimiser(config: Dict) -> keras.optimizers.Optimizer:
  opt_type = config['type'].lower()
  opt_spec = config['spec']

  if opt_type == "sgd":
    return keras.optimizers.SGD(**opt_spec)

  if opt_type == "adam":
    return keras.optimizers.Adam(**opt_spec)

  raise ValueError(f"Unknown optimiser: {config['type']}")

def get_callbacks(out_dir: str) -> None:
  return [
    keras.callbacks.TensorBoard(log_dir=Path(out_dir, "tensorboard")),
    WandbMetricsLogger(log_freq=5),
    WandbModelCheckpoint(Path(out_dir, "models"))
  ]

def train(config: Dict,
          out_dir: str) -> Tuple[keras.callbacks.History, keras.Model]:
  dataset_config = config['dataset']
  train_config = config['training']

  model = create_model(config['architecture'])
  model.compile(optimizer=get_optimiser(train_config['optimiser']),
                loss=train_config['losses'],
                loss_weights=train_config['loss_weights'],
                metrics=train_config['metrics'])

  history = model.fit(create_dataset(config['dataset']),
                      epochs=train_config['epochs'],
                      callbacks=get_callbacks(out_dir),
                      validation_split=train_config['val_split'])

  return history, model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a network from a config.")
  parser.add_argument("config_file", type=str)
  parser.add_argument("--artifact_dir", type=str, default=None)

  args = parser.parse_args()

  with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

  out_dir = args.artifact_dir if args.artifact_dir else "."

  wandb.init(
    project=config['wandb_project'],
    config=config
  )

  keras_history, model = train(config, out_dir)

  wandb.finish()
