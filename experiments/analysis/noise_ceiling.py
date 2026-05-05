import sys
from typing import Callable
import numpy as np
import json
from pathlib import Path

from ..utils import get_CV_trials
from . import rsa

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class NoiseCeiling:
    """
    Compute noise ceiling
    """

    def __init__(
        self,
        list_rois: list,
        list_subjects: list,
        num_folds: int,
        rdm_path_template: Callable,
        noise_ceiling_path: Path,
    ) -> None:

        self.list_rois = list_rois
        self.list_subjects = list_subjects
        self.num_folds = num_folds
        self.rdm_path_template = rdm_path_template
        self.noise_ceiling_path = noise_ceiling_path
        self.sum_rdm = {
            roi: self.sum_rdm_across_subjects(roi) for roi in self.list_rois
        }

    def sum_rdm_across_subjects(self, roi: str):
        """
        Sum fmri rdm data across all subjects for each voxel for a given roi.
        Parameters
        ----------
            roi : str
                The region of interest for which to compute the noise ceiling.
        Returns
        -------
        sum_brain_rdm : np.ndarray
            sumed rdm data with shape (n_trials, n_trials).
        """

        # compute the averaged brain RDM across subjects.
        sum_brain_rdm = None
        for subject in self.list_subjects:
            brain_rdm = np.load(self.rdm_path_template(subject=subject, roi=roi))

            if sum_brain_rdm is None:
                sum_brain_rdm = brain_rdm
            else:
                sum_brain_rdm += brain_rdm

        return sum_brain_rdm

    def average_rdm_across_subjects(self, subject: int, roi: str):
        """
        Average rdm data across subjects except for one test participant data
        Parameters
        ----------
        subject : int
            The test participant whose data is left out when computing the average.
        roi : str
            The region of interest for which to compute the noise ceiling.
        Returns
        -------
        averaged_brain_rdm : np.ndarray
            averaged rdm data with shape (n_trials, n_trials).
        target_brain_rdm : np.ndarray
            The rdm data of the test participant with shape (n_trials, n_trials)
        """
        sum_rdm_this_roi = self.sum_rdm[roi]
        target_brain_rdm = np.load(self.rdm_path_template(subject=subject, roi=roi))
        averaged_brain_rdm = (sum_rdm_this_roi - target_brain_rdm) / (
            len(self.list_subjects) - 1
        )

        return averaged_brain_rdm, target_brain_rdm

    def compute_noise_ceiling(self, roi: str, num_folds: int):
        """
        Compute noise ceiling for each subject by correlating the averaged brain RDM (excluding the test subject) with the test subject's brain RDM.
        Parameters
        ----------
        roi : str
            The region of interest for which to compute the noise ceiling.
        num_folds : int
            The number of folds for cross-validation.
        Returns
        -------
        noise_ceiling : np.ndarray
            An 1D array where each element is the computed noise ceiling for each subject.
        """

        noise_ceiling = np.zeros(len(self.list_subjects))
        for i, subject in enumerate(self.list_subjects):
            averaged_brain_rdm, target_brain_rdm = self.average_rdm_across_subjects(
                subject=subject, roi=roi
            )

            # mimic CV by computing correlation for each fold and average them.
            for fold in range(num_folds):
                cv_trial_indices, _ = get_CV_trials(fold=fold, num_folds=num_folds)
                averaged_brain_rdm_this_fold = averaged_brain_rdm[
                    np.ix_(cv_trial_indices, cv_trial_indices)
                ]
                target_brain_rdm_this_fold = target_brain_rdm[
                    np.ix_(cv_trial_indices, cv_trial_indices)
                ]

                # compute the correlation between the averaged and target RDMs for this fold and store it in noise_ceiling array.
                corr = rsa.correlation_between_rdm(
                    averaged_brain_rdm_this_fold, target_brain_rdm_this_fold
                )
                noise_ceiling[i] += corr

            # average across folds
            noise_ceiling[i] /= num_folds

        return noise_ceiling

    def run(self):
        """Compute noise ceiling for each roi and save the data."""
        noise_ceiling = {}

        # pylint: disable=not-an-iterable
        for roi in self.list_rois:
            nc_data = self.compute_noise_ceiling(roi, self.num_folds)
            noise_ceiling[roi] = (
                nc_data.tolist()
            )  # convert numpy array to list for json serialization

        # save noise ceiling data
        with open(self.noise_ceiling_path, "w", encoding="utf-8") as f:
            json.dump(noise_ceiling, f)
