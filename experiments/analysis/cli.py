"""Module defining the command-line interface for rsa analysis."""

import sys
import argparse
from pathlib import Path
import numpy as np

from . import rsa
from .roi_mask import get_roi_mask
from .noise_ceiling import NoiseCeiling
from .regression import Regression
from .. import utils

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import config


class CommandLineInterface:
    """Represents the command-line interface for rsa analysis."""

    def __init__(self):
        """Initializes a new CommandLineInterface instance."""

        # Initializes some class members
        self.command = None
        self.roi_class = None
        self.list_rois = None
        self.list_subjects = None
        self.num_folds = None
        self.noise_ceiling_path = None
        self.model_rdm_dir_path = None

    def run(self):
        """Runs the command-line interface."""

        # Parses the command line arguments of the application
        self.parse_command_line_arguments()

        # Checks which command the user wants to execute and executes it accordingly
        if self.command == "brain":
            self.brain_rdm()
        elif self.command == "noise-ceiling":
            self.rdm_noise_ceiling()
        elif self.command == "rdm-regression":
            self.rdm_regression()

    def brain_rdm(self):
        """Compute and save rdm from the brain data."""
        if self.list_rois is None or self.list_subjects is None:
            raise ValueError(
                "list_rois and list_subjects must be provided for brain command."
            )
        # pylint: disable=not-an-iterable
        for subject in self.list_subjects:
            fmri_data = np.load(utils.fmri_data_path_template(subject=subject))

            # pylint: disable=not-an-iterable
            for roi in self.list_rois:
                # extract roi data
                mask = get_roi_mask(roi, config.roi_defs_dir_path, self.roi_class)
                roi_data = fmri_data[mask, :].T

                # inputs: samples x voxels
                rdm = rsa.compute_rdm_correlation(roi_data)

                # save data
                np.save(utils.fmri_rdm_path_template(subject=subject, roi=roi), rdm)

    def rdm_noise_ceiling(self):
        """Compute noise ceiling for the regression between model and brain RDMs."""

        nc = NoiseCeiling(
            list_rois=self.list_rois,
            list_subjects=self.list_subjects,
            num_folds=self.num_folds,
            rdm_path_template=utils.fmri_rdm_path_template,
            noise_ceiling_path=self.noise_ceiling_path,
        )

        nc.run()

    def rdm_regression(self):
        """Predict brain rdm from model rdm."""
        reg = Regression(
            list_rois=self.list_rois,
            list_subjects=self.list_subjects,
            model_rdm_dir_path=self.model_rdm_dir_path,
            num_folds=self.num_folds,
        )
        reg.run()

    def parse_command_line_arguments(self):
        """Parses the command line arguments of the application."""

        # Creates a command line argument parser for the application
        argument_parser = argparse.ArgumentParser(
            prog="rdm_interface",
            description="A command line tool for rdm analysis.",
            add_help=True,
        )

        # Adds the commands
        sub_parsers = argument_parser.add_subparsers(dest="command")
        CommandLineInterface.add_brain_command(sub_parsers)
        CommandLineInterface.add_noise_ceiling_command(sub_parsers)
        CommandLineInterface.add_regression_command(sub_parsers)
        # Parses the arguments
        arguments = argument_parser.parse_args()

        if arguments.command is None:
            argument_parser.print_help()
            sys.exit(0)
        self.command = arguments.command
        for key, value in vars(arguments).items():
            setattr(self, key, value)

    @staticmethod
    def add_brain_command(sub_parsers):
        """
        Adds the brain command, which computes RDM from brain data.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """

        brain_command_parser = sub_parsers.add_parser(
            "brain", help="Computes RDM from brain data."
        )

        brain_command_parser.add_argument(
            "--roi_class",
            dest="roi_class",
            type=str,
            default=config.roi_class,
            help="Sets the ROI class. Defaults to config.roi_class.",
        )

        brain_command_parser.add_argument(
            "--list_rois",
            dest="list_rois",
            nargs="+",
            type=str,
            default=config.list_rois,
            help="The list of ROIs for analysis. Defaults to config.list_rois.",
        )

        brain_command_parser.add_argument(
            "--list_subjects",
            dest="list_subjects",
            nargs="+",
            type=str,
            default=config.list_subjects,
            help="The list of subjects for analysis. Defaults to config.list_subjects.",
        )

    @staticmethod
    def add_noise_ceiling_command(sub_parsers):
        """
        Adds the brain command, which computes RDM from brain data.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """

        noise_ceiling_command_parser = sub_parsers.add_parser(
            "noise-ceiling", help="Computes noise ceiling based on brain RDMs."
        )

        noise_ceiling_command_parser.add_argument(
            "--roi_class",
            dest="roi_class",
            type=str,
            default=config.roi_class,
            help="Sets the ROI class. Defaults to config.roi_class.",
        )

        noise_ceiling_command_parser.add_argument(
            "--list_rois",
            dest="list_rois",
            nargs="+",
            type=str,
            default=config.list_rois,
            help="The list of ROIs for analysis. Defaults to config.list_rois.",
        )

        noise_ceiling_command_parser.add_argument(
            "--list_subjects",
            dest="list_subjects",
            nargs="+",
            type=str,
            default=config.list_subjects,
            help="The list of subjects for analysis. Defaults to config.list_subjects.",
        )
        noise_ceiling_command_parser.add_argument(
            "--num_folds",
            dest="num_folds",
            type=int,
            default=config.num_folds,
            help="The number of folds for cross-validation. Defaults to config.num_folds.",
        )
        noise_ceiling_command_parser.add_argument(
            "--noise_ceiling_path",
            dest="noise_ceiling_path",
            type=str,
            default=config.noise_ceiling_path,
            help="The path to save the noise ceiling data. Defaults to config.noise_ceiling_path.",
        )

    @staticmethod
    def add_regression_command(sub_parsers):
        """
        Adds the regression command, which performs regression analysis to predict brain RDM from model RDMs.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """

        regression_command_parser = sub_parsers.add_parser(
            "rdm-regression",
            help="Predicts brain RDM from model RDMs using regression.",
        )

        regression_command_parser.add_argument(
            "--list_rois",
            dest="list_rois",
            nargs="+",
            type=str,
            default=config.list_rois,
            help="The list of ROIs for analysis. Defaults to config.list_rois.",
        )

        regression_command_parser.add_argument(
            "--list_subjects",
            dest="list_subjects",
            nargs="+",
            type=str,
            default=config.list_subjects,
            help="The list of subjects for analysis. Defaults to config.list_subjects.",
        )
        regression_command_parser.add_argument(
            "--num_folds",
            dest="num_folds",
            type=int,
            default=config.num_folds,
            help="The number of folds for cross-validation. Defaults to config.num_folds.",
        )
        regression_command_parser.add_argument(
            "--model_rdm_dir_path",
            dest="model_rdm_dir_path",
            type=str,
            default=config.model_activation_dir_path,
            help="The path to the directory containing model RDM files. Defaults to config.model_rdm_dir_path.",
        )
