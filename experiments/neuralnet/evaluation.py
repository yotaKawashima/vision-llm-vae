"""Module providing functions for evaluating models."""

from nibabel.testing import data_path
import torch
import sys
from pathlib import Path

import torch.nn.functional as F
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluates a model on the full val split and reports average loss metrics.

    Attributes
    - model: torch.nn.Module to be evaluated
    - dataloader: DataLoader that yields evaluation batches
    - logger: Logger for progress and result messages
    """

    def __init__(self, model, dataloader, logger):
        """
        Initializes a instance.

        Parameters
        ----------
            model: torch.nn.Module
                The neural network model to be evaluated
            dataloader: DataLoader
                The DataLoader for evaluation data. Assume no data drop is applied.
            logger: Logger
                The logger that is to used to log the progress of the evaluation.
        """

        self.model = model
        self.model.eval()
        self.dataloader = dataloader
        self.logger = logger

        if not torch.cuda.is_available():
            self.logger.log_error("GPUs are unavailable.")
            sys.exit(1)

        self.device = torch.device("cuda:0")

    def evaluate(
        self,
        model_type: str,
        loss_type: str,
        recon_loss_type: str = "l2",
        llm_alignment_loss_type: str = "cosine_similarity",
        target_alpha: float = None,
        target_beta: float = None,
        target_gamma: float = None,
        target_delta: float = None,
        temperature: float = None,
    ):
        """
        Evaluates the model on the validation dataset.

        Parameters
        ----------
        model_type: str
            model type (Encoder, beta_vae, beta_vae_llm, or decoder)
        loss_type: str
            The type of loss to use
        recon_loss_type: str
            The type of reconstruction loss
        llm_alignment_loss_type: str
            The type of LLM alignment loss
        target_alpha: float
            Alpha value for encoder
        target_beta: float
            Beta value for VAE
        target_gamma: float
            Gamma value for alignment loss
        target_delta: float
            Delta value for llm -> image loss
        temperature: float
            Temperature for soft_nn loss

        Returns
        -------
        dict, list (or None)
            A dictionary containing the average loss metrics over the validation dataset.
            A list of cosine similarity values between text embeddings and image latents for alignment evaluation if applicable, otherwise None.
        """
        self.model.to(self.device)

        self.logger.log_info("Evaluating the trained model...")
        val_num_processed_samples = 0
        loss_metrics = {}
        cosine_sim_list = []
        recon_loss_list = []

        with torch.inference_mode():
            for _, batch in enumerate(self.dataloader):
                # Prepare loss kwargs
                loss_kwargs = self._loss_kwargs(
                    batch,
                    loss_type,
                    model_type,
                    target_alpha,
                    target_beta,
                    recon_loss_type,
                    target_gamma,
                    llm_alignment_loss_type,
                    temperature,
                    target_delta,
                )

                # Forward pass to get loss
                output_dict = self.model.compute_loss(**loss_kwargs)

                # Accumulate loss metrics
                current_batch_size = batch["image"].size(0)
                val_num_processed_samples += current_batch_size
                for k, v in output_dict.items():
                    if k not in loss_metrics:
                        loss_metrics[k] = 0.0
                    loss_metrics[k] += v.item() * current_batch_size

                # keep cosine similarity betwen text embeddings and image latent for alignment evaluation
                if model_type in ["beta_vae", "ae"]:
                    text_embeddings = batch["text_embedding"].to(
                        self.device, non_blocking=True
                    )
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    if model_type == "beta_vae":
                        img_hat, mu, _, _ = self.model(imgs)
                        # pylint: disable=not-callable
                        cosine_sim = F.cosine_similarity(text_embeddings, mu, dim=1)
                    else:  # model_type == "ae"
                        img_hat, latent = self.model(imgs)
                        # pylint: disable=not-callable
                        cosine_sim = F.cosine_similarity(text_embeddings, latent, dim=1)

                    # keep the data for alignment evaluation
                    cosine_sim_list.extend(cosine_sim.detach().cpu().tolist())

                    # also keep the reconstruction loss for alignment evaluation
                    recon_mse_loss = F.mse_loss(img_hat, imgs, reduction="none")
                    # mean over all pixels for each image in the batch
                    recon_mse_loss = recon_mse_loss.view(
                        recon_mse_loss.size(0), -1
                    ).mean(dim=1)
                    recon_loss_list.extend(recon_mse_loss.detach().cpu().tolist())

        # Average loss metrics over all samples
        loss_metrics = {
            k: v / val_num_processed_samples for k, v in loss_metrics.items()
        }

        # Log results
        batch_log = "Evaluation Results:"
        for k, v in loss_metrics.items():
            batch_log += f", {k}: {v:.4f}"

        self.logger.log_success(batch_log)
        self.logger.log_success("Finished evaluating the model")

        if model_type in ["beta_vae", "ae"]:
            alignment_data = {}
            alignment_data["cosine_similarity"] = cosine_sim_list
            alignment_data["reconstruction_mse_loss"] = recon_loss_list
        else:
            alignment_data = None

        return loss_metrics, alignment_data

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
        temperature,
        delta=None,
    ):

        # Tensor types (img, text_embedding) are automatically split and moved to GPUs by DataParallel.
        loss_kwargs = {
            "img": batch["image"].to(self.device, non_blocking=True),
            "loss_type": loss_type,
        }

        # Encoder args
        if model_type == "encoder":
            loss_kwargs.update(
                {
                    "text_embedding": batch["text_embedding"].to(
                        self.device, non_blocking=True
                    ),
                    "alpha": alpha,
                    "temperature": temperature,
                }
            )

        # VAE args
        elif model_type == "beta_vae":
            loss_kwargs.update({"beta": beta, "recon_loss_type": recon_loss_type})

            if "llm_alignment" in loss_type:
                loss_kwargs.update(
                    {
                        "text_embedding": batch["text_embedding"].to(
                            self.device, non_blocking=True
                        ),
                        "gamma": gamma,
                        "llm_alignment_loss_type": llm_alignment_loss_type,
                        "temperature": temperature,
                    }
                )

        # VAE (llm -> image)
        elif model_type in ["beta_vae_llm", "decoder"]:
            loss_kwargs.update(
                {
                    "text_embedding": batch["text_embedding"].to(
                        self.device, non_blocking=True
                    )
                }
            )
            if model_type == "beta_vae_llm" and delta is not None:
                loss_kwargs.update({"delta": delta})

        return loss_kwargs

    def image_reconstruction(
        self,
        model_type,
        fig_dir_path,
        img_mean,
        img_std,
        num_images=50,
        num_images_per_fig=5,
    ):
        """
        Visualizes reconstructed images on the evaluation_data.
        Parameters
        ----------
        model_type: str
            model type (Encoder, beta_vae, beta_vae_llm, or decoder)
        fig_dir_path: str
            path to save the figure showing the reconstructed images
        img_mean: torch.Tensor
            The mean tensor used for normalizing the images during training. Used for denormalizing the images for visualization.
        img_std: torch.Tensor
            The std tensor used for normalizing the images during training. Used for denormalizing the images for visualization.
        num_images: int
            number of images to visualize (starting from the beginning of the dataloader)
        num_images_per_fig: int
            number of images to show in one figure.
        """
        visualized_images = 0
        fig_counter = 0
        dataloader_iter = iter(self.dataloader)
        while visualized_images < num_images:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # If we have exhausted the dataloader, we break out of the loop
                break
            images = batch["image"].to(self.device, non_blocking=True)
            text_embeddings = batch["text_embedding"].to(self.device, non_blocking=True)

            with torch.inference_mode():
                if model_type == "beta_vae":
                    img_hat_coco, _, _, _ = self.model(images)
                elif model_type == "ae":
                    img_hat_coco, _ = self.model(images)
                elif model_type == "decoder":
                    img_hat_coco = self.model(text_embeddings)
                elif model_type == "beta_vae_llm":
                    img_hat_coco, _ = self.model(text_embeddings)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

            # figure name
            fig_name = f"reconstruction_{fig_counter}.svg"
            visualize_images(
                images[:num_images_per_fig],
                img_hat_coco[:num_images_per_fig],
                img_mean,
                img_std,
                "COCO Images: Original vs Reconstructed",
                save_path=fig_dir_path / fig_name,
            )

            # update counter
            visualized_images += num_images_per_fig
            fig_counter += 1

    def store_posterior_var(self, data_path):
        """
        Store the posterior information of the VAE model on the evaluation dataset for visualization.
        """
        posterior_log_var = []
        with torch.inference_mode():
            for _, batch in enumerate(self.dataloader):
                images = batch["image"].to(self.device, non_blocking=True)

                _, log_var = self.model.encode(images)
                posterior_log_var.append(log_var.cpu())

        posterior_log_var = torch.cat(posterior_log_var, dim=0)
        torch.save(posterior_log_var, data_path)
        self.logger.log_success(
            f"Saved posterior log_var of the evaluation dataset to {data_path.absolute()}"
        )


def denormalize(tensor_data, img_mean, img_std):
    return (tensor_data.cpu() * img_std + img_mean).clamp(0, 1)


def visualize_images(
    originals,
    reconstructed,
    img_mean,
    img_std,
    title="Original vs Reconstructed",
    save_path=None,
    show=True,
):
    """
    Compare original and reconstructed images.

    Parameters
    ----------
    originals : torch.Tensor
        Original images
    reconstructed : torch.Tensor
        Reconstructed images
    img_mean_t : torch.Tensor
        The mean tensor used for normalizing the images during training.
    img_std_t : torch.Tensor
        The std tensor used for normalizing the images during training.
    title : str
        Title for the figure
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool
        Whether to display the figure. Default is True.
    """
    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        orig_img = (
            denormalize(originals[i : i + 1], img_mean, img_std)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )
        recon_img = (
            denormalize(reconstructed[i : i + 1], img_mean, img_std)
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .numpy()
        )
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()  # Close figure to free memory
