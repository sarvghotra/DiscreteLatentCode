# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Callbacks to use with the Trainer class and customize the training loop.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm
from transformers.trainer_callback import ExportableState, TrainerCallback
from transformers.trainer_utils import (
    HPSearchBackend,
    IntervalStrategy,
    SaveStrategy,
    has_length,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            When tracking the inputs tokens, the number of tokens seen during training (number of input tokens, not the
            number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
        stateful_callbacks (`List[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved or restored.
            Relevent callbacks should implement a `state` and `from_state` function.
    """

    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None
    stateful_callbacks: List["TrainerCallback"] = None
    train_dataloader_state_dict: Dict[str, Any] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            # We are loading the callbacks in from the state file, no need to process them
            pass
        else:
            # Saveable callbacks get stored as dict of kwargs
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, (ExportableState)):
                    raise TypeError(
                        f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
                    )
                name = callback.__class__.__name__
                if name in stateful_callbacks:
                    # We can have multiple versions of the same callback
                    # if so, we store them as a list of states to restore
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = (
            json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def compute_steps(self, args, max_steps):
        """
        Calculates and stores the absolute value for logging,
        eval, and save steps based on if it was a proportion
        or not.
        """
        for step_kind in ("logging", "eval", "save"):
            num_steps = getattr(args, f"{step_kind}_steps")
            if num_steps is not None:
                if num_steps < 1:
                    num_steps = math.ceil(max_steps * num_steps)
                setattr(self, f"{step_kind}_steps", num_steps)

    def init_training_references(
        self, trainer, train_dataloader, max_steps, num_train_epochs, trial
    ):
        """
        Stores the initial training references needed in `self`
        """
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self, attr, getattr(trainer, attr))

        self.train_dataloader = train_dataloader
        if trainer.hp_name is not None and trainer._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.trial_name = trainer.hp_name(trainer._trial)
        self.trial_params = None
        if trial is not None:
            from transformers.integrations import hp_params

            assignments = (
                trial.assignments
                if trainer.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.trial_params = hp_params(assignments)

        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.is_local_process_zero = trainer.is_local_process_zero()
        self.is_world_process_zero = trainer.is_world_process_zero()
