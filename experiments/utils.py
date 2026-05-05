"""Module providing utility functions."""

from pathlib import Path
import sys
import numpy as np
from typing import Union

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

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


def get_CV_trials(fold: int, num_folds: int = 5, num_trials: int = 515):
    """
    Get the trial indices for a given cv in cross-validation.
    Parameters
    ----------
    fold : int
        The ID of the fold for which to get trial indices.
    num_folds : int
        The number of folds for cross-validation.
    num_trials : int
        The total number of trials in the dataset.
    Returns
    -------
    tuple of lists
        A tuple containing the training and testing trial indices for one fold of cross-validation.
    """

    # Generate trial indices for cross-validation
    trial_indices = np.arange(num_trials)
    fold_size = num_trials // num_folds

    # Set a random seed for reproducibility and shuffle the trial indices
    # pylint: disable=no-member
    random_state = np.random.RandomState(seed=0)
    random_state.shuffle(trial_indices)  # Shuffle the trial indices

    # Get the trial indices for the current fold and the remaining folds
    cv_trial_indices = np.sort(trial_indices[fold * fold_size : (fold + 1) * fold_size])
    cv_trials_other_indices = np.setdiff1d(trial_indices, cv_trial_indices)

    return cv_trial_indices, cv_trials_other_indices


def model_activation_path_template(
    subject: Union[str, int], layer_name: str, split: str, input_modality: str = "image"
):
    """
    Get the path to a model activation file.
    Parameters
    ----------
    subject : Union[str, int]
        The subject ID for which to get the model activation path.
    layer_name : str
        The name of the model layer for which to get the activation path.
    split : str
        The data split for which to get the activation path (e.g., "train" or   "test").
    input_modality : str
        The input modality for which to get the activation path. Defaults to "image".
    Returns
    -------
    Path
        The path to the requested model activation file.
    """
    return config.model_activation_dir_path / (
        f"subj{int(subject):02d}_input_{input_modality}_layer_{layer_name}_"
        + config.checkpoint_path.stem
        + f"_{split}.npy"
    )


def model_rdm_path_template(
    subject: Union[str, int], layer_name: str, input_modality="image"
):
    """
    Get the path to a model RDM file.
    Parameters
    ----------
    subject : Union[str, int]
        The subject ID for which to get the model RDM path.
    layer_name : str
        The name of the model layer for which to get the RDM path.
    input_modality : str
        The input modality for which to get the RDM path. Defaults to "image".
    Returns
    -------
    Path
        The path to the requested model RDM file.
    """
    if subject == "special515":
        path = config.model_activation_dir_path / (
            f"input_{input_modality}_layer_{layer_name}_rdm_special515.npy"
        )
    else:
        path = config.model_activation_dir_path / (
            f"subj{int(subject):02d}_input_{input_modality}_layer_{layer_name}_rdm_NOTspecial515.npy"
        )
    return path


def fmri_data_path_template(subject: Union[str, int]):
    """
    Get the path to a subject's fMRI data file.
    Parameters
    ----------
    subject : Union[str, int]
        The subject ID for which to get the fMRI data path.
    Returns
    -------
    Path
        The path to the requested fMRI data file.
    """
    fmri_file_name = f"subj{int(subject):02d}_betas_average_fsaverage_special515.npy"

    return config.fmri_dir_path / fmri_file_name


def fmri_rdm_path_template(subject: Union[str, int], roi: str):
    """
    Get the path to a subject's fMRI RDM file for a given ROI.
    Parameters
    ----------
    subject : Union[str, int]
        The subject ID for which to get the fMRI RDM path.
    roi : str
        The ROI for which to get the fMRI RDM path.
    Returns
    -------
    Path
        The path to the requested fMRI RDM file.
    """
    path = config.fmri_rdm_dir_path / (
        f"subj{int(subject):02d}_{roi}_rdm_special515.npy"
    )
    return path


def regression_path_template(layer_name: str):
    """
    Get the path to a regression result file for a given layer.
    Parameters
    ----------
    layer_name : str
        The name of the model layer for which to get the regression result path.
    Returns
    -------
    Path
        The path to the requested regression result file.
    """

    path = config.rsa_dir_path / f"rsa_rdm_{layer_name}_regression_special515.json"
    return path
