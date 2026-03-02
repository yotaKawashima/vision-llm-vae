"""Module defining model base."""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn

from .logger import Logger


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    """

    def __init__(self, logger: Optional[Logger] = None) -> None:
        super().__init__()
        self.logger = logger
        self.epoch = 0
        self.optimizer_state_dict = None
        self.lr_scheduler_state_dict = None

    def _log_success(self, content) -> None:
        if self.logger:
            self.logger.log_success(content)
        else:
            print(content)

    def _log_error(self, content) -> None:
        if self.logger:
            self.logger.log_error(content)
            sys.exit(1)

        else:
            raise ValueError(content)

    def _log_info(self, content) -> None:
        if self.logger:
            self.logger.log_info(content)
        else:
            print(content)

    def _set_weights(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> None:
        """
        Set weights of the network. If no checkpoint is provided, initialize using Xavier Uniform initialization.
        """
        # when checkpoint is provided
        if checkpoint_path is not None:
            (
                checkpoint_state_dict,
                self.epoch,
                self.optimizer_state_dict,
                self.lr_scheduler_state_dict,
            ) = self._load_state_dict_from_checkpoint(checkpoint_path)

            self.load_state_dict(checkpoint_state_dict, strict=strict)
            if not strict:
                self.epoch = 0
                self.optimizer_state_dict = None
                self.lr_scheduler_state_dict = None
            self._log_success(f"\nWeights loaded from checkpoint: {checkpoint_path}")

        else:  # when checkpoint is not provided
            self._log_success("Weights initialized")

    def _load_state_dict_from_checkpoint(
        self, checkpoint_path: Union[str, Path]
    ) -> Tuple[
        dict[str, torch.Tensor],
        int,
        Optional[dict[str, torch.Tensor]],
        Optional[dict[str, torch.Tensor]],
    ]:
        """
        Load model weights from a checkpoint file.
        Parameters
        ----------
        checkpoint_path: str or Path
            Path to the checkpoint file.

        Returns
        -------
        Tuple[dict[str, torch.Tensor], int, Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]
            A tuple containing the state dictionary, epoch, optimizer state dictionary, and lr scheduler state dictionary.
        """
        device = next(self.parameters()).device

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # Checkpoint must contain a state_dict.
        # If epoch and optimizer_state_dict are missing, set them to default values.
        _checkpoint_state_dict = checkpoint["state_dict"]
        if (
            "epoch" in checkpoint
            and "optimizer_state_dict" in checkpoint
            and "lr_scheduler_state_dict" in checkpoint
        ):
            epoch = checkpoint["epoch"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            lr_scheduler_state_dict = checkpoint["lr_scheduler_state_dict"]
        else:
            # when optimizer_state_dict does not exist in the checkpoint,
            # the loaded parameters are used as initialization.
            epoch = 0
            optimizer_state_dict = None
            lr_scheduler_state_dict = None

        # remove prefix if exists
        checkpoint_state_dict = {}
        for key, value in _checkpoint_state_dict.items():
            clean_key = key
            if key.startswith("network."):
                clean_key = key.removeprefix("network.")
            elif key.startswith("model."):
                clean_key = key.removeprefix("model.")
            elif key.startswith("module."):
                clean_key = key.removeprefix("module.")
            checkpoint_state_dict[clean_key] = value
        return (
            checkpoint_state_dict,
            epoch,
            optimizer_state_dict,
            lr_scheduler_state_dict,
        )

    @abstractmethod
    def forward(self, inputs: torch.Tensor):
        """
        Forward pass through the network. Given inputs, returns the output.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def compute_loss(self, **kwargs) -> dict:
        """
        Compute the loss for the network. Given inputs, returns the loss.
        """
        raise NotImplementedError("Subclasses must implement the compute_loss method.")
