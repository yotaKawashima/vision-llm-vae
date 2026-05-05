import json
from pathlib import Path
import numpy as np
import re

from .. import utils
from .rsa import correlation_between_rdm


def _natural_key(path):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", path.stem)]


class Regression:
    def __init__(
        self,
        list_subjects: list,
        list_rois: list,
        model_rdm_dir_path: Path,
        num_folds: int = 5,
    ):
        self.list_subjects = list_subjects
        self.list_rois = list_rois
        self.model_rdm_dir_path = model_rdm_dir_path
        self.num_folds = num_folds
        self.num_trials = None  # get this information when loading rdm
        self.encoder_rdms = None
        self.decoder_rdms = None
        self.encoder_layer_names = []
        self.decoder_layer_names = []
        self.mu_rdm = None
        self.latent_rdm = None
        # load model rdms
        self.load_model_rdms()

    def load_model_rdms(self):
        """Load model RDMs from the specified directory."""
        # obtain files in the model RDM directory
        model_rdm_files = sorted(
            self.model_rdm_dir_path.glob("*.npy"), key=_natural_key
        )
        encoder_rdms = []
        decoder_rdms = []

        # load model RDMs
        for file in model_rdm_files:
            # load the rdm and extract the upper triangle
            rdm = np.load(file)
            rdm_upper_triangle = rdm[np.triu_indices_from(rdm, k=1)][:, np.newaxis]

            if self.num_trials is None:
                self.num_trials = rdm.shape[0]

            if "encoder" in file.stem:
                # get the layer name from the file name
                self.encoder_layer_names.append(
                    file.stem.split("encoder.")[1].split("_rdm")[0]
                )
                encoder_rdms.append(rdm_upper_triangle)
            elif "decoder" in file.stem:
                # get the layer name from the file name
                self.decoder_layer_names.append(
                    file.stem.split("decoder.")[1].split("_rdm")[0]
                )
                decoder_rdms.append(rdm_upper_triangle)

            elif "mu" in file.stem:
                self.mu_rdm = rdm_upper_triangle
            elif "latent" in file.stem:
                self.latent_rdm = rdm_upper_triangle
            else:
                raise ValueError(f"Unexpected file name: {file.stem}")

        # keep rdms in the class
        self.encoder_rdms = np.hstack(encoder_rdms)
        self.decoder_rdms = np.hstack(decoder_rdms)

    def mask_fold_indices_cross_validation(self, fold: int):
        """
        Get masks to extract trial indices for the current fold in cross-validation.
        The indices are after extracting the upper triangle of the RDMs.

        Parameters
        ----------
        fold : int
            The current fold number (0-indexed).

        Returns
        -------
        mask : np.ndarray
            A boolean array to extract the trial pairs for the current fold.
        """
        cv_trial_indices, _ = utils.get_CV_trials(
            fold=fold, num_folds=self.num_folds, num_trials=self.num_trials
        )

        row_idx, col_idx = np.triu_indices(self.num_trials, k=1)

        # create a boolean mask to extract the current fold.
        cv_set = set(cv_trial_indices)
        # true if both row and column indices are in the cv_set.
        mask = np.array(
            [(r in cv_set) and (c in cv_set) for r, c in zip(row_idx, col_idx)]
        )

        return mask

    def linear_regression_cv(self, model_rdms: np.ndarray, brain_rdm: np.ndarray):
        """
        Perform linear regression to predict the brain RDM from the model RDMs.

        Parameters
        ----------
        model_rdms : np.ndarray
            A 2D array of shape (num_pairs, num_models) containing the upper triangle of the model RDMs.
        brain_rdm : np.ndarray
            A 1D array of shape (num_pairs,) containing the upper triangle of the brain RDM.

        Returns
        -------
        corr : float
            Average Spearman correlation across folds between predicted and actual brain RDMs.
        weights : np.ndarray
            A 2D array of shape (num_folds, num_models) containing the regression coefficients for each fold.
        """
        corr = 0
        weights = []
        for i_fold in range(self.num_folds):

            # train/test split
            test_mask = self.mask_fold_indices_cross_validation(fold=i_fold)
            train_mask = ~test_mask
            model_rdms_train = model_rdms[train_mask, :]
            model_rdms_test = model_rdms[test_mask, :]
            brain_rdm_train = brain_rdm[train_mask]
            brain_rdm_test = brain_rdm[test_mask]

            # estimate regression coefficients using least squares
            c, _, _, _ = np.linalg.lstsq(model_rdms_train, brain_rdm_train, rcond=None)

            # evaluate the regression model on the test (spearman)
            brain_rdm_pred = model_rdms_test @ c
            corr += correlation_between_rdm(
                brain_rdm_test, brain_rdm_pred, apply_upper_triangle=False
            )

            weights.append(c)

        # average correlation across folds
        corr /= self.num_folds
        weights = np.vstack(weights)

        return corr, weights

    def run(self):
        """Run regression for all layer groups, ROIs, and subjects. Results stored in self.results."""
        for layers_name in ["all", "encoder", "decoder", "mu", "latent"]:
            if layers_name == "all":
                model_rdms = np.hstack((self.encoder_rdms, self.decoder_rdms))
            elif layers_name == "encoder":
                model_rdms = self.encoder_rdms
            elif layers_name == "decoder":
                model_rdms = self.decoder_rdms
            elif layers_name == "mu":
                model_rdms = self.mu_rdm
            elif layers_name == "latent":
                model_rdms = self.latent_rdm
            else:
                raise ValueError(f"Unexpected layers_name: {layers_name}")

            corr_dict = {}
            # pylint: disable=not-an-iterable
            for roi in self.list_rois:
                corr = np.zeros(len(self.list_subjects))
                # pylint: disable=not-an-iterable
                for i, subject in enumerate(self.list_subjects):
                    brain_rdm = np.load(
                        utils.fmri_rdm_path_template(subject=subject, roi=roi)
                    )
                    brain_rdm_upper_triangle = brain_rdm[
                        np.triu_indices_from(brain_rdm, k=1)
                    ]

                    corr[i], _ = self.linear_regression_cv(
                        model_rdms, brain_rdm_upper_triangle
                    )

                # store corr data in dict.
                corr_dict[roi] = corr.tolist()

            # save noise ceiling data
            regression_path = utils.regression_path_template(layers_name)
            with open(regression_path, "w", encoding="utf-8") as f:
                json.dump(corr_dict, f)
