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
from typing import Dict, List, Tuple

import yaml
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import keras
from keras import Model
from keras.optimizers import Adam, Optimizer, SGD

from dataset_factory import create_dataset
from model_factory import create_model

from models_lib.utils.data import split_dataset


def get_optimiser(config: Dict) -> Optimizer:
    opt_type = config["type"].lower()
    opt_spec = config["spec"]

    if opt_type == "sgd":
        return SGD(**opt_spec)

    if opt_type == "adam":
        return Adam(**opt_spec)

    raise ValueError(f"Unknown optimiser: {config['type']}")


def get_callbacks(
    out_dir: str, use_tensorboard: bool = False, use_wandb: bool = False
) -> List[keras.callbacks.Callback]:
    callbacks = []

    if use_tensorboard:
        callbacks += [keras.callbacks.TensorBoard(log_dir=Path(out_dir, "tensorboard"))]

    if use_wandb:
        callbacks += [
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath=Path(out_dir, "models")),
        ]

    return callbacks


def train(
    config: Dict,
    out_dir: str,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    pretrain: bool = False,
) -> Tuple[keras.callbacks.History, Model]:
    train_config = config["training"]

    pretrain_grid_shape = (8, 8) if pretrain else None

    m = create_model(config["architecture"], pretraining_grid_shape=pretrain_grid_shape)
    m.compile(
        optimizer=get_optimiser(train_config["optimiser"]),
        loss=train_config["losses"],
        loss_weights=train_config["loss_weights"],
        metrics=train_config["metrics"],
    )

    ds = create_dataset(config["dataset"], pretraining_grid_shape=pretrain_grid_shape)
    if isinstance(ds, tuple):
        if "val_split" in train_config:
            raise ValueError(
                "val_split cannot be specified when using an already split tfds."
            )

        train_ds, val_ds = ds
    else:
        if "val_split" not in train_config:
            raise ValueError("val_split must be specified when using a non split tfds.")

        val_split = train_config["val_split"]

        train_ds, val_ds = split_dataset(ds, val_split)

    m.summary()

    history = m.fit(
        train_ds,
        epochs=train_config["epochs"],
        callbacks=get_callbacks(
            out_dir, use_tensorboard=use_tensorboard, use_wandb=use_wandb
        ),
        validation_data=val_ds,
    )

    return history, m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network from a config.")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument("--wandb", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--tensorboard", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--pretraining", default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)

    if args.wandb:
        if args.pretrain:
            cfg["wandb"]["name"] += "_pretrain"
        wandb.init(**cfg["wandb"], config=cfg)

    keras_history, model = train(
        cfg,
        args.artifact_dir if args.artifact_dir else ".",
        args.pretrain,
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        pretrain=args.pretrain,
    )

    if args.wandb:
        wandb.finish()
