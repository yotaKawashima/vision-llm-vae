"""Module defining the command-line interface for VAE experiments."""

import sys
import argparse
import json
import torch
from pathlib import Path

from . import models
from . import models_resnet
from .logger import ConsoleLogger
from .datasets import (
    ApplyTransformSubset,
    CocoTextEmbeddingImageDataset,
    CocoH5Dataset,
)  # , NSDStimulusDataset
from .training import DataParallelismTrainer

# from .evaluation import Evaluator


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import config

# from experiments.neuralnet.activation_extraction import ActivationExtractor


class CommandLineInterface:
    """Represents the command-line interface."""

    def __init__(self):
        """Initializes a new CommandLineInterface instance."""

        # Initializes the logger
        self.logger = ConsoleLogger()

        # Initializes some class members
        self.command = None
        self.resnet_flag = None
        self.batch_size = None
        self.learning_rate = None
        self.number_of_epochs = None
        self.model_type = None
        self.loss_type = None
        self.recon_loss_type = None
        self.llm_alignment_loss_type = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.clip_grad_norm = None
        self.model = None
        self.target_layers = None
        self.vision_bias = None
        self.input_modality = None
        self.writer_path = None
        self.num_workers = None
        self.encoder_checkpoint = None
        self.ae_checkpoint = None
        self.vae_checkpoint = None

    def run(self):
        """Runs the command-line interface."""

        # Parses the command line arguments of the application
        self.parse_command_line_arguments()
        model_lib = models_resnet if self.resnet_flag else models
        if self.resnet_flag:
            self.logger.log_info("Using ResNet-based model architecture.")

        if self.model_type == "encoder":
            self.model = model_lib.Encoder(
                latent_dim=config.latent_dim,
                image_size=config.img_resize,
                checkpoint_path=config.checkpoint_path,
                logger=self.logger,
            )
        elif self.model_type == "ae":
            if self.encoder_checkpoint and config.checkpoint_path is None:
                raise ValueError("encoder_checkpoint=True requires checkpoint_path to be set.")
            self.model = model_lib.AE(
                latent_dim=config.latent_dim,
                image_size=config.img_resize,
                checkpoint_path=config.checkpoint_path,
                logger=self.logger,
                set_weights=config.checkpoint_path is not None,
                encoder_checkpoint=self.encoder_checkpoint,
            )
        elif self.model_type == "beta_vae":
            if self.ae_checkpoint and config.checkpoint_path is None:
                raise ValueError("ae_checkpoint=True requires checkpoint_path to be set.")
            self.model = model_lib.BetaVAE(
                latent_dim=config.latent_dim,
                image_size=config.img_resize,
                checkpoint_path=config.checkpoint_path,
                logger=self.logger,
                set_weights=config.checkpoint_path is not None,
                ae_checkpoint=self.ae_checkpoint,
            )
        elif self.model_type == "beta_vae_llm":
            self.model = model_lib.BetaVAEScalingLLM(
                latent_dim=config.latent_dim,
                image_size=config.img_resize,
                checkpoint_path=config.checkpoint_path,
                logger=self.logger,
                vae_checkpoint=self.vae_checkpoint,
            )
        else:
            self.logger.log_error(f"Unknown model type '{self.model_type}' specified.")
            sys.exit(1)

        # Checks which command the user wants to execute and executes it accordingly
        if self.command == "train":
            self.train()
        elif self.command == "evaluation":
            self.evaluation()
        elif self.command == "activation-extraction":
            self.extract_activations(input_modality=self.input_modality)

    def train(self):
        """Trains the model."""
        # Original coco datasets
        if config.coco_version == "Doerig":
            # Doerig et al Nat Mach Intell dataset
            train_dataset = CocoH5Dataset(
                h5_path=config.coco_doerig_h5_path,
                split="train",
                embedding_key="all_mpnet_base_v2_mean_embeddings",
                img_transform=config.img_transform_train_h5,
                logger=self.logger,
            )
            val_dataset = CocoH5Dataset(
                h5_path=config.coco_doerig_h5_path,
                split="val",
                embedding_key="all_mpnet_base_v2_mean_embeddings",
                img_transform=config.img_transform_val_h5,
                logger=self.logger,
            )
        else:
            # train model
            dataset = CocoTextEmbeddingImageDataset(
                split="train",
                img_transform=None,
            )
            val_size = 5000  # Use a fixed number of samples for validation
            train_size = len(dataset) - val_size

            # randomly split the dataset into train and val splits with a fixed random seed for reproducibility
            indices = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(0)
            ).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Subset the dataset for training and validation
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)

            # use data augmentaiton for training data and no augmentation for validation data
            train_dataset = ApplyTransformSubset(
                train_subset, config.img_transform_train
            )
            val_dataset = ApplyTransformSubset(val_subset, config.img_transform_val)

        # data loader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            prefetch_factor=2,
        )
        val_num_workers = min(
            4, int(len(val_dataset) // self.batch_size)
        )  # Use fewer workers for validation
        val_num_workers = max(1, val_num_workers)  # Ensure at least one worker
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        trainer = DataParallelismTrainer(
            self.model, train_dataloader, self.logger, self.writer_path
        )

        if config.training_history_path.exists():
            with open(config.training_history_path, "r", encoding="utf-8") as f:
                initial_history = json.load(f)
            self.logger.log_info(
                f"Loaded existing history from {config.training_history_path}"
            )
        else:
            initial_history = None

        history_dict = trainer.train(
            val_dataloader,
            self.learning_rate,
            self.number_of_epochs,
            self.model_type,
            self.loss_type,
            self.recon_loss_type,
            self.llm_alignment_loss_type,
            self.alpha,
            self.beta,
            self.gamma,
            self.clip_grad_norm,
            initial_history=initial_history,
        )

        # save training history (dicts of list (floats))
        with open(config.training_history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)
        self.logger.log_success(
            f"Saved training history to {config.training_history_path}"
        )

    def evaluation(self):
        """Evaluates the model."""
        raise NotImplementedError(
            "Evaluation for h5 dataset not implemented yet. You need text embeddings for the dataset."
        )
        # # evaluate the model on the val split (no data drop for evaluation)
        # if config.coco_version == "Doerig":
        #     # Doerig et al Nat Mach Intell dataset
        #     test_dataset = CocoH5Dataset(
        #         h5_path=config.coco_doerig_h5_path,
        #         split="test",
        #         embedding_key="all_mpnet_base_v2_mean_embeddings",
        #         img_transform=config.img_transform_h5,
        #     )
        # else:
        #     # use val dataset from the original coco dataset for evaluation
        #     test_dataset = CocoTextEmbeddingImageDataset(
        #         split="val",
        #         img_transform=config.img_transform,
        #     )

        # dataloader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        # evaluator = Evaluator(self.model, dataloader, self.logger)
        # evaluator.evaluate()

    def extract_activations(self, input_modality="image"):
        """Extracts activations from the model on the dataset."""
        raise NotImplementedError()
        # # train data for each subject
        # for subject in config.subjects:
        #     dataset = NSDStimulusDataset(subject=subject)

        #     dataloader = torch.utils.data.DataLoader(
        #         dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True
        #     )

        #     # Extracts activations
        #     extractor = ActivationExtractor(
        #         self.model,
        #         dataloader,
        #         self.logger,
        #         self.target_layers,
        #         config.model_activation_path,
        #     )
        #     extractor.extract(input_modality=input_modality, vision_bias=self.vision_bias)

        # # Test data
        # dataset = NSDStimulusDataset(subject="test")
        # dataloader = torch.utils.data.DataLoader(
        #     dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True
        # )
        # # Extracts activations
        # extractor = ActivationExtractor(
        #     self.model,
        #     dataloader,
        #     self.logger,
        #     self.target_layers,
        #     config.model_activation_path,
        # )
        # extractor.extract(input_modality=input_modality, vision_bias=self.vision_bias)

    def parse_command_line_arguments(self):
        """Parses the command line arguments of the application."""

        # Creates a command line argument parser for the application
        argument_parser = argparse.ArgumentParser(
            prog="vae_interface",
            description="A command line tool for training models and extracting features.",
            add_help=True,
        )

        # Adds the command line argument for the version of the application
        argument_parser.add_argument(
            "-V",
            "--verbosity",
            dest="verbosity",
            type=str,
            choices=["none", "error", "success", "info"],
            default="info",
            help='Sets the verbosity level of the logging. Defaults to "info".',
        )

        # Adds the commands
        sub_parsers = argument_parser.add_subparsers(dest="command")
        CommandLineInterface.add_training_command(sub_parsers)
        CommandLineInterface.add_evaluation_command(sub_parsers)
        CommandLineInterface.add_extract_activations_command(sub_parsers)

        # Parses the arguments
        arguments = argument_parser.parse_args()

        if arguments.command is None:
            argument_parser.print_help()
            sys.exit(0)

        self.command = arguments.command
        self.logger.set_verbosity(arguments.verbosity)
        for key, value in vars(arguments).items():
            if key != "verbosity":
                setattr(self, key, value)

    @staticmethod
    def add_training_command(sub_parsers):
        """
        Adds the training command, which trains a neural network.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """

        train_command_parser = sub_parsers.add_parser(
            "train", help="Trains neural networks."
        )
        train_command_parser.add_argument(
            "--resnet-flag",
            dest="resnet_flag",
            action=argparse.BooleanOptionalAction,
            default=config.resnet_flag,
            help="If set, use the ResNet-based model architecture. Defaults to config.resnet_flag.",
        )
        train_command_parser.add_argument(
            "--num_workers",
            dest="num_workers",
            type=int,
            default=config.num_workers,
            help="The number of worker processes to use for data loading. Defaults to 24.",
        )
        train_command_parser.add_argument(
            "-e",
            "--number-of-epochs",
            dest="number_of_epochs",
            type=int,
            default=config.number_of_epochs,
            help="The number of epochs to train for. Defaults to config.number_of_epochs.",
        )
        train_command_parser.add_argument(
            "-b",
            "--batch-size",
            dest="batch_size",
            type=int,
            default=config.batch_size,
            help="The size of the mini-batch used during training and testing. Defaults to config.batch_size.",
        )
        train_command_parser.add_argument(
            "--model-type",
            dest="model_type",
            type=str,
            default=config.model_type,
            help="Sets the model type. Defaults to config.model_type.",
        )
        train_command_parser.add_argument(
            "--loss-type",
            dest="loss_type",
            type=str,
            default=config.loss_type,
            help="The type of loss to use. Defaults to config.loss_type.",
        )
        train_command_parser.add_argument(
            "--recon-loss-type",
            dest="recon_loss_type",
            type=str,
            default=config.recon_loss_type,
            help="The type of reconstructionloss to use. Defaults to config.recon_loss_type.",
        )
        train_command_parser.add_argument(
            "--llm-alignment-loss-type",
            dest="llm_alignment_loss_type",
            type=str,
            default=config.llm_alignment_loss_type,
            help="The type of LLM alignment loss to use. Defaults to config.llm_alignment_loss_type.",
        )
        train_command_parser.add_argument(
            "-l",
            "--learning-rate",
            dest="learning_rate",
            type=float,
            default=config.learning_rate,
            help="The learning rate used in the training of the model. Defaults to config.learning_rate.",
        )
        train_command_parser.add_argument(
            "-A",
            "--alpha",
            dest="alpha",
            type=float,
            default=config.alpha,
            help="The weight for cosine similarity loss for the encoder. Defaults to config.alpha.",
        )
        train_command_parser.add_argument(
            "-B",
            "--beta",
            dest="beta",
            type=float,
            default=config.beta,
            help="The beta value for the beta-VAE loss function. Defaults to config.beta.",
        )
        train_command_parser.add_argument(
            "--clip-grad-norm",
            dest="clip_grad_norm",
            type=float,
            default=config.clip_grad_norm,
            help="The maximum norm for gradient clipping. Defaults to None (no clipping).",
        )
        train_command_parser.add_argument(
            "-G",
            "--gamma",
            dest="gamma",
            type=float,
            default=config.gamma,
            help="The weight for llm direction alignment loss. Defaults to config.gamma.",
        )
        train_command_parser.add_argument(
            "--writer-path",
            dest="writer_path",
            type=str,
            default=config.writer_path,
            help="The path for TensorBoard logs. Defaults to config.writer_path.",
        )
        train_command_parser.add_argument(
            "--encoder-checkpoint",
            dest="encoder_checkpoint",
            action=argparse.BooleanOptionalAction,
            default=config.encoder_checkpoint,
            help="The path to the encoder checkpoint to initialize the encoder with (only applicable for ae model). Defaults to config.encoder_checkpoint.",
        )
        train_command_parser.add_argument(
            "--ae-checkpoint",
            dest="ae_checkpoint",
            action=argparse.BooleanOptionalAction,
            default=config.ae_checkpoint,
            help="The path to the ae checkpoint to initialize the ae model with (only applicable for beta_vae model). Defaults to config.ae_checkpoint.",
        )
        train_command_parser.add_argument(
            "--vae-checkpoint",
            dest="vae_checkpoint",
            action=argparse.BooleanOptionalAction,
            default=config.vae_checkpoint,
            help="The path to the vae checkpoint to initialize the vae model with (only applicable for beta_vae model). Defaults to config.vae_checkpoint.",
        )

    @staticmethod
    def add_evaluation_command(sub_parsers):
        """
        Adds the evaluation command, which evaluates a trained model.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """
        evaluation_command_parser = sub_parsers.add_parser(
            "evaluation", help="Evaluates a trained model."
        )
        evaluation_command_parser.add_argument(
            "--resnet-flag",
            dest="resnet_flag",
            action=argparse.BooleanOptionalAction,
            default=config.resnet_flag,
            help="If set, use the ResNet-based model architecture. Defaults to config.resnet_flag.",
        )
        evaluation_command_parser.add_argument(
            "--num_workers",
            dest="num_workers",
            type=int,
            default=config.num_workers,
            help="The number of worker processes to use for data loading. Defaults to 24.",
        )
        evaluation_command_parser.add_argument(
            "-b",
            "--batch-size",
            dest="batch_size",
            type=int,
            default=config.batch_size,
            help="The size of the mini-batch used during evaluation. Defaults to config.batch_size.",
        )
        evaluation_command_parser.add_argument(
            "--model-type",
            dest="model_type",
            type=str,
            default=config.model_type,
            help="Sets the model type. Defaults to config.model_type.",
        )

    @staticmethod
    def add_extract_activations_command(sub_parsers):
        """
        Adds the extract-activations command, which extracts activations from a trained model.

        Parameters
        ----------
        sub_parsers: Action
            The sub parsers to which the command is to be added.
        """
        extract_activations_command_parser = sub_parsers.add_parser(
            "activation-extraction", help="Extracts activations from a trained model."
        )
        extract_activations_command_parser.add_argument(
            "--resnet-flag",
            dest="resnet_flag",
            action=argparse.BooleanOptionalAction,
            default=config.resnet_flag,
            help="If set, use the ResNet-based model architecture. Defaults to config.resnet_flag.",
        )
        extract_activations_command_parser.add_argument(
            "--num_workers",
            dest="num_workers",
            type=int,
            default=config.num_workers,
            help="The number of worker processes to use for data loading. Defaults to 24.",
        )
        extract_activations_command_parser.add_argument(
            "-b",
            "--batch-size",
            dest="batch_size",
            type=int,
            default=config.batch_size,
            help="The size of the mini-batch used during activation extraction. Defaults to config.batch_size.",
        )
        extract_activations_command_parser.add_argument(
            "--model-type",
            dest="model_type",
            type=str,
            default=config.model_type,
            help="Sets the model type. Defaults to config.model_type.",
        )
        extract_activations_command_parser.add_argument(
            "-t",
            "--target-layers",
            dest="target_layers",
            nargs="+",
            type=str,
            default=config.target_layers,
            help="The target layers for activation extraction. Defaults to config.target_layers.",
        )
        extract_activations_command_parser.add_argument(
            "--vision-bias",
            dest="vision_bias",
            type=float,
            default=config.vision_bias,
            help="Bias toward vision when combining text embedings and visual latent values (only for input_modality=both). Defaults to config.vision_bias.",
        )
        extract_activations_command_parser.add_argument(
            "--input-modality",
            dest="input_modality",
            type=str,
            default=config.input_modality,
            help="The modality of input data to the model when extracting model activations. Defaults to config.input_modality.",
        )
