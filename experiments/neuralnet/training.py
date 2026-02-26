"""Module providing functions for training models."""

import sys
import math
from pathlib import Path
from typing import Union, Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter

from .logger import Logger

project_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_path))

from experiments.utils import make_checkpoint_path


class TrainingWrapper(torch.nn.Module):
    """
    DataParallel only parallelizes the 'forward' method, so this wrapper
    forwards the forward call to model.compute_loss.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model.compute_loss(**kwargs)


class DataParallelismTrainer:
    """
    Trainer that performs multi-GPU training using PyTorch's DataParallel.
    Supports Encoder, AE, and VAE training phases dynamically.

    This class wraps a torch.nn.Module with torch.nn.DataParallel so each incoming
    batch is split across multiple GPUs and processed in parallel. Gradients
    are aggregated on the master device (cuda:0) and parameters are updated
    there.

    Notes
    - Assumes 4 GPUs are available for training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        logger: Logger,
        writer_path: Path,
    ):
        """
        Initializes a new instance.

        Parameters
        ----------
        model: torch.nn.Module
            The neural network model to be trained
        dataloader: DataLoader
            The DataLoader for training data
        logger: Logger
            The logger for logging training progress
        writer_path: Path
            The path for TensorBoard logs

        """
        self.original_model = model
        self.dataloader = dataloader
        self.logger = logger
        self.optimizer = None
        self.scheduler = None
        # self.model will be wrapped with DataParallel inside the train method
        self.model = None
        self.writer = SummaryWriter(log_dir=writer_path)
        self.logger.log_info(f"TensorBoard logging to: {writer_path}")

    def train(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        learning_rate: float,
        number_of_epochs: int,
        model_type: str,
        loss_type: str,
        recon_loss_type: str = "l2",
        llm_alignment_loss_type: str = "cosine_similarity",
        target_alpha: float = 1.0,
        target_beta: float = 1.0,
        target_gamma: float = 1.0,
        clip_grad_norm: float = 1.0,
        initial_history: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Trains the model dynamically handling Encoder, AE, and VAE outputs.

        Parameters
        ----------
        val_dataloader: DataLoader
            The DataLoader for validation data
        learning_rate: float
            The learning rate
        number_of_epochs: int
            The number of epochs
        model_type: str
            model (Encoder, AE, or VAE)
        loss_type: str
            The type of loss to use ("standard", "llm_prior", or "llm_alignment")
        recon_loss_type: str
            The type of reconstruction loss to use ("l2" or "l1")
        llm_alignment_loss_type: str
            The type of LLM alignment loss to use ("cosine_similarity" or "l2")
        target_alpha: float
            The target alpha value for cosine similarity weighting (only used for encoder)
        target_beta: float
            The target beta value for KL divergence weighting. (the value at the end of warmup)
        target_gamma: float
            The target gamma value for alignment loss (only used if loss_type is "llm_alignment")
        clip_grad_norm: float or None
            The maximum norm for gradient clipping. If None, no clipping is applied.
        initial_history: dict or None
            An optional dictionary containing initial training history to continue from. If None, a new history will be created.
        Returns
        -------
        dict
            A dictionary containing training loss histories

        """
        if not torch.cuda.is_available():
            self.logger.log_error("GPUs are unavailable.")
            sys.exit(1)
        gpu_count = torch.cuda.device_count()
        if gpu_count != 4:
            self.logger.log_error(
                "4 GPUs are required for data parallelism in training."
            )
            sys.exit(1)

        # Gets the first CUDA device and use it as the main device.
        master_device = torch.device("cuda:0")

        # 1. Move the original model to the master device
        self.original_model.to(master_device)

        # 2. Create the wrapper and apply DataParallel
        # This ensures that calls to self.model(...) are parallelized
        training_wrapper = TrainingWrapper(self.original_model)
        self.model = torch.nn.DataParallel(
            training_wrapper, device_ids=list(range(gpu_count))
        )
        self.model.to(master_device)

        # 3. Setup Optimizer
        # Pass parameters of the original model, not the wrapper
        params_to_optimize = [
            p for p in self.original_model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize, lr=learning_rate, weight_decay=1e-2
        )

        # 4. Load previous training state (if available)
        if self.original_model.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.original_model.optimizer_state_dict)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=number_of_epochs, eta_min=1e-6
        )

        if self.original_model.lr_scheduler_state_dict is not None:
            self.scheduler.load_state_dict(self.original_model.lr_scheduler_state_dict)
            # Update T_max in case number_of_epochs has changed
            self.scheduler.T_max = number_of_epochs

        # Restore Epoch
        start_epoch = self.original_model.epoch
        warmup_epochs = max(1, number_of_epochs // 3)

        history = initial_history if initial_history is not None else {}

        for epoch in range(start_epoch + 1, number_of_epochs + 1):
            self.model.train()  # Ensure model is in training mode for each batch
            # --- Beta Annealing ---
            if model_type == "beta_vae":
                if epoch < warmup_epochs:
                    progress = epoch / warmup_epochs
                    annealing_ratio = 0.5 * (1 - math.cos(math.pi * progress))
                else:
                    annealing_ratio = 1.0

                beta = max(target_beta * annealing_ratio, 1e-4)
                gamma = target_gamma
            else:
                beta = None
                gamma = None

            if model_type == "encoder":
                alpha = target_alpha  # For encoder, we can also apply a similar annealing strategy to alpha if desired. Here we keep it constant.
            else:
                alpha = None

            epoch_metrics = {}
            num_processed_samples = 0

            for batch_step, batch in enumerate(self.dataloader):
                # Prepare input data for DataParallel
                loss_kwargs = self._loss_kwargs(
                    batch,
                    loss_type,
                    model_type,
                    alpha,
                    beta,
                    recon_loss_type,
                    gamma,
                    llm_alignment_loss_type,
                )

                self.optimizer.zero_grad()

                # --- Forward Pass (Parallel Execution) ---
                # self.model(...) -> DataParallel -> Wrapper -> compute_loss
                # Each value in output_dict is a Tensor of size [num_gpus]
                output_dict = self.model(**loss_kwargs)

                # --- Average the Loss ---
                # Average the losses from each GPU to get a single scalar
                loss = output_dict["loss"].mean()
                loss.backward()

                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.original_model.parameters(), max_norm=clip_grad_norm
                    )
                self.optimizer.step()

                # Accumulate Metrics
                current_batch_size = batch["image"].size(0)
                num_processed_samples += current_batch_size

                for k, v in output_dict.items():
                    # Average the values returned from each GPU and aggregate
                    val = v.mean().item()
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0.0
                    epoch_metrics[k] += val * current_batch_size

                # Logging (every 10 batches)
                if (batch_step + 1) % 10 == 0:
                    log_msg = f"Epoch: {epoch}, Batch: {batch_step + 1}"
                    # Add all metrics dynamically
                    for k, v in epoch_metrics.items():
                        avg = v / num_processed_samples
                        log_msg += f", {k}: {avg:.4f}"

                    # Add Alpha info if relevant
                    if model_type == "encoder":
                        log_msg += f", alpha: {alpha:.4f}"

                    # Add Beta/Gamma info if relevant
                    if model_type == "beta_vae":
                        log_msg += f", beta: {beta:.4f}"
                        if "llm_alignment" in loss_type:
                            log_msg += f", gamma: {gamma:.4f}"

                    self.logger.log_info(log_msg)

            self.scheduler.step()

            # End of Epoch Logging
            batch_log = f"Epoch: {epoch}"
            for k, v in epoch_metrics.items():
                avg = v / num_processed_samples
                batch_log += f", {k}: {avg:.4f}"
                # for TensorBoard
                self.writer.add_scalar(f"Loss/{k}", avg, epoch)

                # Update History
                if k not in history:
                    history[k] = []
                history[k].append(avg)

            if model_type == "encoder":
                # message and history for alpha
                batch_log += f", alpha: {alpha:.4f}"
                if "alpha" not in history:
                    history["alpha"] = []
                history["alpha"].append(alpha)

            if model_type == "beta_vae":
                # message and history for beta
                batch_log += f", beta: {beta:.4f}"
                if "beta" not in history:
                    history["beta"] = []
                history["beta"].append(beta)

                if "llm_alignment" in loss_type:
                    # message and history for gamma
                    batch_log += f", gamma: {gamma:.4f}"
                    if "gamma" not in history:
                        history["gamma"] = []
                    history["gamma"].append(gamma)

            self.logger.log_success(batch_log)

            val_num_processed_samples = 0
            val_epoch_metrics = {}
            # validation
            self.model.eval()  # Set model to evaluation mode for validation
            with torch.no_grad():
                for batch_step, batch in enumerate(val_dataloader):
                    # Prepare input data for DataParallel
                    # Tensor types (img, text_embedding) are automatically split and moved to GPUs by DataParallel.
                    loss_kwargs = self._loss_kwargs(
                        batch,
                        loss_type,
                        model_type,
                        alpha,
                        beta,
                        recon_loss_type,
                        gamma,
                        llm_alignment_loss_type,
                    )

                    # --- Forward Pass ---
                    output_dict = self.model(**loss_kwargs)

                    # Accumulate Metrics
                    current_batch_size = batch["image"].size(0)
                    val_num_processed_samples += current_batch_size
                    for k, v in output_dict.items():
                        # Average the values returned from each GPU and aggregate
                        validation_key = f"val_{k}"
                        if validation_key not in val_epoch_metrics:
                            val_epoch_metrics[validation_key] = 0.0
                        val_epoch_metrics[validation_key] += (
                            v.mean().item() * current_batch_size
                        )

                # log for validation
                batch_log = f"Validation: Epoch: {epoch}"
                for k, v in val_epoch_metrics.items():
                    avg = v / val_num_processed_samples
                    batch_log += f", {k}: {avg:.4f}"
                    # Update History
                    if k not in history:
                        history[k] = []
                    history[k].append(avg)
                    # for TensorBoard
                    metric_name = k.replace("val_", "")
                    self.writer.add_scalar(f"Loss/{metric_name}_val", avg, epoch)

                # show validation results in the same log message
                self.logger.log_success(batch_log)

            # Save Checkpoint
            self.original_model.epoch = epoch  # Update internal epoch
            self.writer.flush()
            if epoch % 5 == 0:
                path = self.save_model_checkpoint(epoch)
                self.logger.log_success(f"Saved checkpoint to {path}")

        # Reports that the training has finished
        self.writer.close()
        self.logger.log_info("Saving trained model...")
        path = self.save_model_checkpoint(self.original_model.epoch)
        self.logger.log_success(f"Finished training. Saved model to {path}.")

        history.update(
            {"start_epoch": start_epoch + 1, "end_epoch": self.original_model.epoch}
        )
        return history

    def _loss_kwargs(
        self,
        batch,
        loss_type,
        model_type,
        alpha,
        beta,
        recon_loss_type,
        gamma,
        llm_alignment_loss_type,
    ):

        # Tensor types (img, text_embedding) are automatically split and moved to GPUs by DataParallel.
        loss_kwargs = {"img": batch["image"], "loss_type": loss_type}

        # Encoder args
        if model_type == "encoder":
            loss_kwargs.update(
                {"text_embedding": batch["text_embedding"], "alpha": alpha}
            )

        # VAE args
        elif model_type == "beta_vae":
            loss_kwargs.update({"beta": beta, "recon_loss_type": recon_loss_type})

            if "llm_alignment" in loss_type:
                loss_kwargs.update(
                    {
                        "text_embedding": batch["text_embedding"],
                        "gamma": gamma,
                        "llm_alignment_loss_type": llm_alignment_loss_type,
                    }
                )

        # VAE (llm -> image)
        elif model_type == "beta_vae_llm":
            loss_kwargs.update({"text_embedding": batch["text_embedding"]})

        return loss_kwargs

    def save_model_checkpoint(self, epoch: Union[int, str]) -> Path:
        """
        Saves the model checkpoint to disk.

        Parameters
        ----------
        epoch: int
            The current epoch number

        Returns
        -------
        Path
            The path to the saved checkpoint file
        """

        # Generates the file name and appends it to the path
        path = make_checkpoint_path(epoch)

        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.original_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        return path
