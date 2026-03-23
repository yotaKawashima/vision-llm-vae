"""Module defining Variational Autoencoder (VAE) models."""

from typing import Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import Logger
from .models_base import BaseModel


class Encoder(BaseModel):
    """
    Standalone encoder: maps images to latent vectors.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        image_size: int = 128,
        checkpoint_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
        set_weights: bool = True,
    ) -> None:

        super().__init__(logger=logger)

        self._build_encoder(in_channels, latent_dim, image_size)

        if set_weights:
            self._set_weights(checkpoint_path=checkpoint_path)

    def _build_encoder(
        self, in_channels: int, latent_dim: int, image_size: int
    ) -> None:
        """Build encoder layers. Called by __init__ and AE.__init__."""
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.hidden_dims = [32, 64, 128, 256, 512]

        s = image_size
        self.encoder = nn.Sequential(
            self.make_encoder_block(
                in_channels, 32, kernel_size=7, padding=3, spatial_size=s
            ),
            self.make_encoder_block(32, 32, kernel_size=7, padding=3, spatial_size=s),
            self.make_encoder_block(32, 64, kernel_size=5, padding=2, spatial_size=s),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_encoder_block(
                64, 64, kernel_size=5, padding=2, spatial_size=(s := s // 2)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_encoder_block(
                64, 128, kernel_size=3, padding=1, spatial_size=(s := s // 2)
            ),
            self.make_encoder_block(128, 128, kernel_size=3, padding=1, spatial_size=s),
            self.make_encoder_block(128, 256, kernel_size=3, padding=1, spatial_size=s),
            self.make_encoder_block(256, 256, kernel_size=3, padding=1, spatial_size=s),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_encoder_block(
                256, 512, kernel_size=1, padding=0, spatial_size=(s := s // 2)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.make_encoder_block(
                512, 512, kernel_size=1, padding=0, spatial_size=s // 2
            ),
            nn.AdaptiveAvgPool2d(2),
        )

        # compute shape by doing one forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            out = self.encoder(dummy)
            self.encoder_output_shape = out.shape[1:]
            self.output_features = out.flatten(1).shape[1]

        self.fc_mu = nn.Linear(self.output_features, latent_dim)

    @staticmethod
    def make_encoder_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        spatial_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.ReLU(),
            nn.LayerNorm([out_channels, spatial_size, spatial_size]),
        )

    def _freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        for param in self.fc_mu.parameters():
            param.requires_grad = not freeze
        self._encoder_frozen = freeze
        if freeze:
            self.encoder.eval()
            self.fc_mu.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_encoder_frozen", False):
            self.encoder.eval()
            self.fc_mu.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → latent code.

        Parameters
        ----------
        inputs: torch.Tensor (batch x C x H x W)
        Returns
        -------
        torch.Tensor (batch x latent_dim)
        """
        return self.encode(inputs)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to a latent vector.

        Parameters
        ----------
        img: torch.Tensor (batch x C x H x W)
        Returns
        -------
        torch.Tensor (batch x latent_dim)
        """
        intermediate = self.encoder(img)
        intermediate = torch.flatten(intermediate, start_dim=1)
        return self.fc_mu(intermediate)

    @staticmethod
    def soft_nn_loss(
        mu: torch.Tensor,
        text_emb: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Soft nearest-neighbour (InfoNCE-style) contrastive loss.

        Parameters
        ----------
        mu: torch.Tensor (batch x latent_dim)
        text_emb: torch.Tensor (batch x text_embedding_dim)
        temperature: float
        Returns
        -------
        torch.Tensor
        """
        sim = mu @ text_emb.T / temperature  # [batch, batch]
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def latent_alignment_loss(
        self,
        latent_variables: torch.Tensor,
        text_emb: torch.Tensor,
        llm_alignment_loss_type: str = "norm_and_cosine_similarity",
        temperature: float | None = 0.1,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compute alignment loss between latent variables and text embeddings.

        Parameters
        ----------
        latent_variables: torch.Tensor (batch x latent_dim)
        text_emb: torch.Tensor (batch x text_embedding_dim)
        llm_alignment_loss_type: str
        temperature: float or None
            Temperature for soft_nn loss. Required when llm_alignment_loss_type is "soft_nn" or "norm_and_soft_nn".
        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        if llm_alignment_loss_type == "norm_and_cosine_similarity":
            # pylint: disable=not-callable
            cosine_sim_loss = (
                1 - F.cosine_similarity(latent_variables, text_emb, dim=1).mean()
            )
            norms = latent_variables.norm(p=2, dim=1)
            norm_loss = F.mse_loss(norms, torch.ones_like(norms), reduction="mean")
            return cosine_sim_loss, norm_loss
        elif llm_alignment_loss_type == "norm_and_cosine_similarity_smoothL1":
            # pylint: disable=not-callable
            cosine_sim_loss = (
                1 - F.cosine_similarity(latent_variables, text_emb, dim=1).mean()
            )
            norms = latent_variables.norm(p=2, dim=1)
            norm_loss = F.smooth_l1_loss(
                norms, torch.ones_like(norms), reduction="mean", beta=1.0
            )
            return cosine_sim_loss, norm_loss
        elif llm_alignment_loss_type == "cosine_similarity":
            # pylint: disable=not-callable
            return 1 - F.cosine_similarity(latent_variables, text_emb, dim=1).mean()
        elif llm_alignment_loss_type == "soft_nn":
            return self.soft_nn_loss(
                latent_variables, text_emb, temperature=temperature
            )
        elif llm_alignment_loss_type == "norm_and_soft_nn":
            soft_nn_loss = self.soft_nn_loss(
                latent_variables, text_emb, temperature=temperature
            )
            norms = latent_variables.norm(p=2, dim=1)
            norm_loss = F.mse_loss(norms, torch.ones_like(norms), reduction="mean")
            return soft_nn_loss, norm_loss
        elif llm_alignment_loss_type == "l1":
            return F.l1_loss(latent_variables, text_emb, reduction="mean")
        elif llm_alignment_loss_type == "l2":
            return F.mse_loss(latent_variables, text_emb, reduction="mean")
        else:
            raise ValueError(
                f"Unknown loss_type for llm alignment: {llm_alignment_loss_type}"
            )

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute encoder alignment loss.

        Parameters
        ----------
        **kwargs:
            img : torch.Tensor (batch x C x H x W)
            text_embedding: torch.Tensor (batch x text_embedding_dim)
            loss_type: str  (default "norm_and_cosine_similarity")
            alpha: float    (default 1.0)
            temperature: float  (required for "soft_nn" and "norm_and_soft_nn")
        Returns
        -------
        dict
        """
        img = kwargs.get("img")
        text_embedding = kwargs.get("text_embedding")
        loss_type = kwargs.get("loss_type", "norm_and_cosine_similarity")
        alpha = kwargs.get("alpha", 1.0)
        temperature = kwargs.get("temperature", None)
        if loss_type in ["soft_nn", "norm_and_soft_nn"] and temperature is None:
            raise ValueError(
                "temperature must be provided for soft_nn llm_alignment_loss_type."
            )
        latent_variable = self.forward(img)
        output = self.latent_alignment_loss(
            latent_variable,
            text_embedding,
            llm_alignment_loss_type=loss_type,
            temperature=temperature,
        )

        if loss_type in [
            "norm_and_cosine_similarity",
            "norm_and_cosine_similarity_smoothL1",
        ]:
            cosine_sim_loss, norm_loss = output
            total_loss = alpha * cosine_sim_loss + norm_loss
            return {
                "loss": total_loss,
                "cosine_sim_loss": cosine_sim_loss,
                "norm_loss": norm_loss,
            }
        elif loss_type == "norm_and_soft_nn":
            soft_nn_loss, norm_loss = output
            total_loss = alpha * soft_nn_loss + norm_loss
            return {
                "loss": total_loss,
                "soft_nn_loss": soft_nn_loss,
                "norm_loss": norm_loss,
            }
        else:
            return {"loss": output}


class Decoder(BaseModel):
    """
    Standalone decoder: maps latent vectors (e.g. LLM embeddings) directly to images.
    This is the base class for all image-generating models.
    """

    # Fixed by encoder architecture: AdaptiveAvgPool2d(2) with 512 output channels.
    # All subclasses share this bottleneck shape.
    _ENC_OUT_CHANNELS: int = 512
    _ENC_OUT_SPATIAL: int = 2

    def __init__(
        self,
        latent_dim: int = 768,
        image_size: int = 128,
        checkpoint_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
        set_weights: bool = True,
    ) -> None:
        super().__init__(logger=logger)

        self._build_decoder(latent_dim, image_size)

        if set_weights:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path must be provided if set_weights is True."
                )
            self._set_weights(checkpoint_path=checkpoint_path)

    def _build_decoder(self, latent_dim: int, image_size: int) -> None:
        """Build decoder layers. Called by __init__ and AE.__init__."""
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder_output_shape: Tuple[int, int, int] = (
            self._ENC_OUT_CHANNELS,
            self._ENC_OUT_SPATIAL,
            self._ENC_OUT_SPATIAL,
        )
        self.output_features: int = (
            self._ENC_OUT_CHANNELS * self._ENC_OUT_SPATIAL * self._ENC_OUT_SPATIAL
        )

        self.decoder_input = nn.Linear(latent_dim, self.output_features)
        s_d = image_size // 16
        group_size = 4
        self.decoder = nn.Sequential(
            nn.Upsample(size=(s_d, s_d), mode="bilinear"),
            self.make_decoder_block(
                512, 512, kernel_size=3, padding=1, num_groups=512 // group_size
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            self.make_decoder_block(
                512, 256, kernel_size=3, padding=1, num_groups=256 // group_size
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            self.make_decoder_block(
                256, 256, kernel_size=3, padding=1, num_groups=256 // group_size
            ),
            self.make_decoder_block(
                256, 128, kernel_size=3, padding=1, num_groups=128 // group_size
            ),
            self.make_decoder_block(
                128, 128, kernel_size=3, padding=1, num_groups=128 // group_size
            ),
            self.make_decoder_block(
                128, 64, kernel_size=3, padding=1, num_groups=64 // group_size
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            self.make_decoder_block(
                64, 64, kernel_size=5, padding=2, num_groups=64 // group_size
            ),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            self.make_decoder_block(
                64, 32, kernel_size=5, padding=2, num_groups=32 // group_size
            ),
            self.make_decoder_block(
                32, 32, kernel_size=7, padding=3, num_groups=32 // group_size
            ),
            nn.Conv2d(32, 3, kernel_size=7, padding=3, padding_mode="reflect"),
        )

    @staticmethod
    def make_decoder_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        num_groups: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(),
        )

    def decode(self, latent_variables: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables into a reconstructed image.

        Parameters
        ----------
        latent_variables: torch.Tensor (batch x latent_dim)
        Returns
        -------
        torch.Tensor (batch x 3 x H x W)
        """
        output = self.decoder_input(latent_variables)
        output = output.view(-1, *self.encoder_output_shape)
        return self.decoder(output)

    def reconstruction_loss(
        self, img: torch.Tensor, img_hat: torch.Tensor, recon_loss_type: str = "l2"
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between target and predicted image.

        Parameters
        ----------
        img : torch.Tensor (batch x channel x height x width)
        img_hat : torch.Tensor (batch x channel x height x width)
        recon_loss_type: str  "l1" or "l2"
        Returns
        -------
        torch.Tensor
        """
        if recon_loss_type == "l1":
            return F.l1_loss(img, img_hat, reduction="mean")
        elif recon_loss_type == "l2":
            return F.mse_loss(img, img_hat, reduction="mean")
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: LLM embedding → reconstructed image.

        Parameters
        ----------
        inputs: torch.Tensor (batch x latent_dim)
        Returns
        -------
        torch.Tensor (batch x 3 x H x W)
        """
        return self.decode(inputs)

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute reconstruction loss for standalone decoder training (LLM → image).

        Parameters
        ----------
        **kwargs:
            img: torch.Tensor (batch x channel x height x width)
                Target image
            text_embedding: torch.Tensor (batch x latent_dim)
                LLM embedding used as input
            loss_type: str  (default "l2")
        Returns
        -------
        dict
        """
        img = kwargs.get("img")
        text_embedding = kwargs.get("text_embedding")
        loss_type = kwargs.get("loss_type", "l2")
        img_hat = self.forward(text_embedding)
        return {
            "loss": self.reconstruction_loss(img, img_hat, recon_loss_type=loss_type)
        }


class AE(Encoder, Decoder):
    """
    Autoencoder: combines Encoder and Decoder.
    Input flow: image → encode → decode → reconstructed image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        image_size: int = 128,
        checkpoint_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
        set_weights: bool = True,
        encoder_checkpoint: bool = False,
    ) -> None:
        """Initialize the Autoencoder (AE) model."""
        # Initialize BaseModel (nn.Module) exactly once, bypassing Encoder/Decoder __init__
        BaseModel.__init__(self, logger=logger)

        # Build encoder (sets self.encoder, self.fc_mu, self.encoder_output_shape, etc.)
        self._build_encoder(in_channels, latent_dim, image_size)

        # Build decoder (sets self.decoder_input, self.decoder)
        # encoder_output_shape and output_features are already set above with the same values
        self._build_decoder(latent_dim, image_size)

        # set weights
        if set_weights:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path must be provided if set_weights is True."
                )
            strict = not encoder_checkpoint
            self._set_weights(checkpoint_path=checkpoint_path, strict=strict)
            # Freeze encoder weights
            self._freeze_encoder(freeze=True)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: image → (reconstructed image, latent code).

        Parameters
        ----------
        inputs: torch.Tensor (batch x C x H x W)
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Reconstructed image, latent code
        """
        latent = self.encode(inputs)
        return self.decode(latent), latent

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute image reconstruction loss.

        Parameters
        ----------
        **kwargs:
            img : torch.Tensor (batch x C x H x W)
            loss_type: str  (default "l2")
        Returns
        -------
        dict
        """
        img = kwargs.get("img")
        loss_type = kwargs.get("loss_type", "l2")
        img_hat = self.forward(img)[0]
        return {
            "loss": self.reconstruction_loss(img, img_hat, recon_loss_type=loss_type)
        }


class BetaVAE(AE):
    """
    Beta Variational Autoencoder (Beta-VAE) model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        image_size: int = 128,
        checkpoint_path: Union[str, Path] = None,
        logger: Optional[Logger] = None,
        set_weights: bool = True,
        ae_checkpoint: bool = False,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            image_size=image_size,
            checkpoint_path=None,
            logger=logger,
            set_weights=False,  # We will set weights (or initialize them) after adding fc_logvar
        )

        # Add a fully connected layer for log variance
        self.fc_logvar = nn.Linear(self.output_features, latent_dim)

        # set weights
        if set_weights:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path must be provided if set_weights is True."
                )
            strict = not ae_checkpoint
            self._set_weights(checkpoint_path=checkpoint_path, strict=strict)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var).

        Parameters
        ----------
        mu: pytorch.Tensor (batch x latent_dim)
            Mean of the latent Gaussian
        log_var: pytorch.Tensor (batch x latent_dim)
            Log variance of the latent Gaussian

        Returns
        -------
        pytorch.Tensor (batch x latent_dim)
            Reparameterized sample
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Provide inputs to the encoder network and returns the latent codes.
        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to encoder

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of latent codes (mu and log_var)
        """
        # image feature from encoder
        intermediate = self.encoder(img)
        intermediate = torch.flatten(intermediate, start_dim=1)

        # compute mu and log_var
        mu = self.fc_mu(intermediate)
        log_var = self.fc_logvar(intermediate)
        return (mu, log_var)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network. Given inputs, returns the reconstructed image.

        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to the network
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructed image, mean and log variance of the latent Gaussian
        """
        mu, log_var = self.encode(inputs)
        latent_variables = self.reparameterize(mu, log_var)
        return self.decode(latent_variables), mu, log_var, latent_variables

    def kl_divergence_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss normalized by dimension size.

        Parameters
        ----------
        mu: pytorch.Tensor (batch x latent_dim)
            Mean of the latent Gaussian
        log_var: pytorch.Tensor (batch x latent_dim)
            Log variance of the latent Gaussian
        Returns
        -------
        pytorch.Tensor
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = 0.5 * torch.sum(-1 - log_var + log_var.exp() + mu.pow(2))

        # Return after normalizing by batch size
        return kl_loss / mu.size(0)

    def compute_vae_loss(
        self,
        img: torch.Tensor,
        beta: float,
        recon_loss_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the standard VAE.
        Parameters
        ----------
        img : pytorch.Tensor (batch x channel x height x width)
            Input image tensor to the network
        beta: float
            Weight for the KL divergence term
        recon_loss_type: str
            Type of reconstruction loss to compute.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Total loss, reconstruction loss, KL divergence loss
        """
        img_hat, mu, log_var, _ = self.forward(img)
        recon_loss = self.reconstruction_loss(
            img,
            img_hat,
            recon_loss_type=recon_loss_type,
        )
        kl_loss = self.kl_divergence_loss(mu, log_var)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def compute_vae_loss_with_alignment(
        self,
        img: torch.Tensor,
        beta: float,
        recon_loss_type: str,
        gamma: float,
        text_embedding: torch.Tensor,
        llm_alignment_loss_type: str = "cosine_similarity",
        temperature: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the VAE after pretraining.
        image reconstruction loss, kl loss, llm reconstruction loss

        Parameters
        ----------
        img : pytorch.Tensor (batch x channel x height x width)
            Input image tensor to the network
        beta: float
            Weight for the KL divergence term
        recon_loss_type: str
            Type of reconstruction loss to compute.
        gamma : float
            Weight for the LLM alignment term.
        text_embedding: torch.Tensor (batch x text_embedding_dim)
            Text embedding tensor to condition the network
        llm_alignment_loss_type: str
            Type of LLM alignment loss to compute.
        temperature: float or None
            Temperature for soft_nn alignment loss. Required when llm_alignment_loss_type is "soft_nn".
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Total loss, reconstruction loss, KL divergence loss, alignment loss
        """
        # image reconstruction from image
        img_hat, mu, log_var, _ = self.forward(img)

        # reconstruction loss
        recon_loss = self.reconstruction_loss(
            img,
            img_hat,
            recon_loss_type,
        )

        # kl loss
        kl_loss = self.kl_divergence_loss(
            mu,
            log_var,
        )

        # llm allignment loss (direction)
        if llm_alignment_loss_type not in [
            "cosine_similarity",
            "soft_nn",
        ]:
            raise ValueError(
                "Supported llm alignment loss types for BetaVAE: cosine_similarity, soft_nn."
            )
        llm_alignment_loss = self.latent_alignment_loss(
            mu,
            text_embedding,
            llm_alignment_loss_type=llm_alignment_loss_type,
            temperature=temperature,
        )

        total_loss = recon_loss + beta * kl_loss + gamma * llm_alignment_loss

        return total_loss, recon_loss, kl_loss, llm_alignment_loss

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute the loss based on the specified loss type.

        Parameters
        ----------
        **kwargs
            img : torch.Tensor
                Input image tensor.
            loss_type : str
                Type of loss to compute: "standard", "llm_alignment".
            beta : float
                Weight for the KL divergence term.
            recon_loss_type : str
                Type of reconstruction loss to compute. (default is "l2")
            llm_alignment_loss_type: str
                Type of LLM alignment loss to compute. (required for "llm_alignment")
            gamma : float
                Weight for the LLM alignment term (required for "llm_alignment").
            text_embedding : torch.Tensor
                Text embedding tensor (required for "llm_alignment").
            temperature : float
                Temperature for soft_nn alignment loss (required when llm_alignment_loss_type is "soft_nn").
        Returns
        -------
        dict
            Loss returned by the respective loss function.
        """
        img = kwargs["img"]
        loss_type = kwargs["loss_type"]
        beta = kwargs["beta"]
        recon_loss_type = kwargs.get("recon_loss_type", "l2")
        llm_alignment_loss_type = kwargs.get("llm_alignment_loss_type", None)
        gamma = kwargs.get("gamma", None)
        text_embedding = kwargs.get("text_embedding", None)
        temperature = kwargs.get("temperature", None)

        if loss_type == "standard":
            total_loss, recon_loss, kl_loss = self.compute_vae_loss(
                img,
                beta,
                recon_loss_type,
            )
            return {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
            }
        elif loss_type == "llm_alignment":
            if (
                llm_alignment_loss_type is None
                or gamma is None
                or text_embedding is None
            ):
                raise ValueError(
                    "gamma, text_embedding and llm_alignment_loss_type must be provided for llm_alignment loss."
                )
            if llm_alignment_loss_type == "soft_nn" and temperature is None:
                raise ValueError(
                    "temperature must be provided for soft_nn llm_alignment_loss_type."
                )
            total_loss, recon_loss, kl_loss, llm_alignment_loss = (
                self.compute_vae_loss_with_alignment(
                    img,
                    beta,
                    recon_loss_type,
                    gamma,
                    text_embedding,
                    llm_alignment_loss_type=llm_alignment_loss_type,
                    temperature=temperature,
                )
            )
            return {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "llm_alignment_loss": llm_alignment_loss,
            }
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


class BetaVAEScalingLLM(BetaVAE):
    """
    Beta Variational Autoencoder (Beta-VAE) model with scaling .
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        image_size: int = 128,
        checkpoint_path: Union[str, Path] = None,
        logger: Optional[Logger] = None,
        vae_checkpoint: bool = False,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            image_size=image_size,
            checkpoint_path=None,
            logger=logger,
            set_weights=False,  # We will set weights (or initialize them) after adding llm_scaler
        )

        # Add a fully connected layer for log variance
        self.llm_scaler = nn.Linear(latent_dim, 1)

        # set weights
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided.")
        strict = not vae_checkpoint
        self._set_weights(checkpoint_path=checkpoint_path, strict=strict)
        # Freeze encoder weights
        self._freeze_vae(freeze=True)

    def scale(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale inputs

        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to the network
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Scaled inputs, scaling factor
        """
        scaling_factor = self.llm_scaler(inputs)
        return inputs * scaling_factor, scaling_factor

    def _freeze_vae(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the vae weights.

        Parameters
        ----------
        freeze: bool
            If True, freeze the encoder weights. If False, unfreeze them.
        """
        self._freeze_encoder(freeze=freeze)
        self._freeze_decoder(freeze=freeze)
        self._vae_frozen = freeze

    def _freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the encoder weights of vae.
        Parameters
        ----------
        freeze: bool
            If True, freeze the encoder weights. If False, unfreeze them.
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        for param in self.fc_mu.parameters():
            param.requires_grad = not freeze
        for param in self.fc_logvar.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.encoder.eval()
            self.fc_mu.eval()
            self.fc_logvar.eval()

    def _freeze_decoder(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the decoder weights.
        Parameters
        ----------
        freeze: bool
            If True, freeze the decoder weights. If False, unfreeze them.
        """
        for param in self.decoder_input.parameters():
            param.requires_grad = not freeze
        for param in self.decoder.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.decoder_input.eval()
            self.decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_vae_frozen", False):
            self.encoder.eval()
            self.fc_mu.eval()
            self.fc_logvar.eval()
            self.decoder_input.eval()
            self.decoder.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network. Given inputs, returns the reconstructed image.

        Parameters
        ----------
        inputs: pytorch.Tensor (batch x latent_dim)
            Input latent tensor to the network
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Reconstructed image, scaling factor
        """
        scaled_inputs, scaling_factor = self.scale(inputs)
        return self.decode(scaled_inputs), scaling_factor

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute the loss based on the specified loss type.

        Parameters
        ----------
        **kwargs
            img : torch.Tensor
                Input image tensor.
            text_embedding : torch.Tensor
                Text embedding tensor
            loss_type : str
                Type of loss to compute. (default is l2)
        Returns
        -------
        dict
            Loss returned by the respective loss function.
        """
        img = kwargs.get("img")
        text_embedding = kwargs.get("text_embedding")
        loss_type = kwargs.get("loss_type", "l2")
        img_hat, _ = self.forward(text_embedding)
        return {"loss": self.reconstruction_loss(img, img_hat, loss_type)}
