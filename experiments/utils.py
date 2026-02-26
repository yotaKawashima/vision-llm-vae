"""Module providing utility functions."""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from typing import Union
import config


def make_checkpoint_path(epoch: Union[int, str]) -> Path:
    """
    Make the path to a training checkpoint file.

    Parameters
    ----------
    epoch : int
    Returns
    -------
    Path
        The filesystem path to the requested checkpoint file.
    """

    config.coco_checkpoints_dir_path.mkdir(parents=True, exist_ok=True)

    return config.coco_checkpoints_dir_path / f"checkpoint_epoch{epoch}.ckpt"


def get_last_checkpoint_path(dir_path: Path) -> Union[Path, None]:
    """
    Get the path to a training checkpoint file.

    Parameters
    ----------
    dir_path : Path
        The directory path where checkpoints are stored.

    Returns
    -------
    Path or None
        The filesystem path to the requested checkpoint file.
    """

    # if checkpoint exists, get the last checkpoint file in the directory
    checkpoint_files = list(dir_path.glob(f"checkpoint_epoch*.ckpt"))
    if checkpoint_files:
        # sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    else:  # if no checkpoint exists, return None
        return None


def replace_subdir(path: Path, old: str, new: str) -> Path:
    """
    Replace a subdirectory in a given path.

    Parameters
    ----------
    path : Path
        The original filesystem path.
    old : str
        The subdirectory name to be replaced.
    new : str
        The new subdirectory name.

    Returns
    -------
    Path
        The modified filesystem path with the subdirectory replaced.
    """
    parts = list(path.parts)
    # Replace the old subdirectory with the new one
    new_parts = [new if part == old else part for part in parts]

    return Path(*new_parts)
