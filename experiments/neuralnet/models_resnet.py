"""Module defining Variational Autoencoder (VAE) models."""

from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from .logger import Logger
from .models_base import BaseModel


class Encoder(BaseModel):

    def __init__(
        self,
        in_channels: int = 3,  # backwards compatibility, not actually used since ResNet18 always takes 3-channel input
        latent_dim: int = 768,
        image_size: int = 224,
        checkpoint_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
        set_weights: bool = True,
    ) -> None:

        assert (
            in_channels == 3
        ), "Currently only in_channels=3 is supported since we are using ResNet18 as the encoder backbone."
        super().__init__(logger=logger)

        self._build_encoder(in_channels, latent_dim, image_size)

        # set weights
        if set_weights:
            self._set_weights(checkpoint_path=checkpoint_path)

    def _build_encoder(
        self,
        in_channels: int,  # backwards compatibility, not actually used since ResNet18 always takes 3-channel input
        latent_dim: int,
        image_size: int,
    ) -> None:
        """Build encoder layers. Called by __init__ and AE.__init__."""
        self.latent_dim = latent_dim
        self.image_size = image_size

        assert (
            image_size % 32 == 0
        ), f"image_size must be divisible by 32, got {image_size}"
        stem_size = (
            image_size // 32
        )  # spatial size after layer4 (e.g. 128→4, 224→7, 256→8)

        # backbone without avgpool (output: 512 x stem_size x stem_size)
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.encoder_output_shape = (512, stem_size, stem_size)
        self.output_features = 512 * stem_size * stem_size

        # Latent space mapping
        self.fc_mu = nn.Linear(self.output_features, latent_dim)

    def _freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the encoder weights.

        Parameters
        ----------
        freeze: bool
            If True, freeze the encoder weights. If False, unfreeze them.
        """
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
        Forward pass through the network. Given inputs, returns the latent code.

        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to the network
        Returns
        -------
        torch.Tensor
            Latent code
        """
        return self.encode(inputs)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Provide inputs to the encoder network and returns the latent codes.
        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to encoder

        Returns
        -------
        torch.Tensor
            Latent code
        """
        # image feature from encoder
        intermediate = self.encoder(img)
        intermediate = torch.flatten(intermediate, start_dim=1)
        return self.fc_mu(intermediate)

    def latent_alignment_loss(
        self,
        latent_variables: torch.Tensor,
        text_emb: torch.Tensor,
        llm_alignment_loss_type: str = "norm_and_cosine_similarity",
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compute the loss for the AE.

        Parameters
        ----------
        latent_variables: pytorch.Tensor (batch x latent_dim)
            Latent variables tensor from the network
        text_emb: pytorch.Tensor (batch x text_embedding_dim)
            Text embedding tensor to align with latent variables
        llm_alignment_loss_type: str
            Type of reconstruction loss to compute.
        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            Computed loss
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
        Compute the loss for the encoder.

        Parameters
        ----------
        **kwargs:
            img : pytorch.Tensor (batch x channel x height x width)
                Input image tensor to the network
            text_embedding: torch.Tensor (batch x text_embedding_dim)
                Text embedding tensor to condition the network
            loss_type: str
                Type of reconstruction loss to compute. (default is "norm_and_cosine_similarity")
            alpha: float
                The weight for cosine similarity loss for the encoder. (default is 1.0)
        Returns
        -------
        dict
            Computed loss
        """
        img = kwargs.get("img")
        text_embedding = kwargs.get("text_embedding")
        loss_type = kwargs.get("loss_type", "norm_and_cosine_similarity")
        alpha = kwargs.get("alpha", 1.0)
        latent_variable = self.forward(img)
        output = self.latent_alignment_loss(
            latent_variable, text_embedding, llm_alignment_loss_type=loss_type
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
        else:
            return {"loss": output}


class DecoderBasicBlock(nn.Module):
    """
    Decoder block mirroring ResNet18's BasicBlock, with optional ×2 upsampling.
    Mirrors the 3×3 → 3×3 structure with a residual skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
    ) -> None:
        super().__init__()

        # 3×3 spatial (with optional ×2 upsample)
        if upsample:
            self.conv1: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3×3 spatial
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection: adjust spatial/channels when needed
        if upsample or in_channels != out_channels:
            skip_layers: List[nn.Module] = []
            if upsample:
                skip_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            skip_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
            skip_layers.append(nn.BatchNorm2d(out_channels))
            self.skip: nn.Module = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

    @staticmethod
    def make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        upsample_first: bool = False,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            DecoderBasicBlock(in_channels, out_channels, upsample=upsample_first)
        ]
        for _ in range(1, num_blocks):
            layers.append(DecoderBasicBlock(out_channels, out_channels, upsample=False))
        return nn.Sequential(*layers)


class Decoder(BaseModel):
    """
    Standalone decoder: maps latent vectors (e.g. LLM embeddings) directly to images.
    This is the base class for all image-generating models.
    """

    def __init__(
        self,
        in_channels: int = 3,  # kept for API consistency
        latent_dim: int = 768,
        image_size: int = 224,
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
        # Spatial size after ResNet18 layer4: image_size // 32
        stem_size = image_size // 32
        self.encoder_output_shape: Tuple[int, int, int] = (512, stem_size, stem_size)
        self.output_features: int = 512 * stem_size * stem_size

        # latent → 512×(s×s) → 256×(2s×2s) → 128×(4s×4s) → 64×(8s×8s) → 64×(8s×8s) → 3×image_size²
        self.decoder_input = nn.Linear(latent_dim, self.output_features)
        self.decoder = nn.Sequential(
            # mirror layer4 (2 blocks): 512 → 256, spatial ×2
            DecoderBasicBlock.make_layer(512, 256, num_blocks=2, upsample_first=True),
            # mirror layer3 (2 blocks): 256 → 128, spatial ×2
            DecoderBasicBlock.make_layer(256, 128, num_blocks=2, upsample_first=True),
            # mirror layer2 (2 blocks): 128 → 64, spatial ×2
            DecoderBasicBlock.make_layer(128, 64, num_blocks=2, upsample_first=True),
            # mirror layer1 (2 blocks): 64 → 64, no spatial change
            DecoderBasicBlock.make_layer(64, 64, num_blocks=2, upsample_first=False),
            # reverse stem: maxpool(stride=2) → conv1(7×7, stride=2, 64→3)
            nn.Upsample(scale_factor=2, mode="nearest"),  # reverse maxpool
            nn.Upsample(scale_factor=2, mode="nearest"),  # reverse conv1 stride=2
            nn.Conv2d(64, 3, kernel_size=7, padding=3),  # reverse conv1
        )

    def decode(self, latent_variables: torch.Tensor) -> torch.Tensor:
        """
        Provide latent variables to the decoder network and returns the reconstructed image.

        Parameters
        ----------
        latent_variables: pytorch.Tensor (batch x latent_dim)
            Latent variables to decode

        Returns
        -------
        pytorch.Tensor
            Reconstructed image tensor
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
    Autoencoder: combines ResNet18 Encoder and Decoder.
    Input flow: image → encode → decode → reconstructed image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        image_size: int = 224,
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
            if checkpoint_path is not None:
                if encoder_checkpoint:
                    # Case 1: load pretrained encoder checkpoint, then train decoder only.
                    # strict=False because the checkpoint has no decoder keys.
                    self._set_weights(checkpoint_path=checkpoint_path, strict=False)
                    self._freeze_encoder(freeze=True)
                else:
                    # load pretrained AE checkpoint, then fine-tune entire AE.
                    self._set_weights(checkpoint_path=checkpoint_path, strict=True)
            else:
                # Case 2: no checkpoint provided, initialize entire AE with random weights.
                self._log_info(
                    "No checkpoint provided, initializing entire AE (ResNet18 with IMAGENET1K_V1)."
                )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network. Given inputs, returns the reconstructed image.

        Parameters
        ----------
        img: pytorch.Tensor (batch x Channel x Height x Width)
            Input image tensor to the network
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Reconstructed image, latent code
        """
        latent = self.encode(inputs)
        return self.decode(latent), latent

    def compute_loss(self, **kwargs) -> dict:
        """
        Compute the loss for the AE.

        Parameters
        ----------
        **kwargs:
            img : pytorch.Tensor (batch x channel x height x width)
                Input image tensor to the network
            loss_type: str
                Type of reconstruction loss to compute. (default is "l2")
        Returns
        -------
        dict
            Computed loss
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
        image_size: int = 224,
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
        assert llm_alignment_loss_type in [
            "cosine_similarity"
        ], "Currently only cosine similarity loss is supported for llm alignment loss."
        llm_alignment_loss = self.latent_alignment_loss(
            mu,
            text_embedding,
            llm_alignment_loss_type=llm_alignment_loss_type,
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
            total_loss, recon_loss, kl_loss, llm_alignment_loss = (
                self.compute_vae_loss_with_alignment(
                    img,
                    beta,
                    recon_loss_type,
                    gamma,
                    text_embedding,
                    llm_alignment_loss_type=llm_alignment_loss_type,
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
        image_size: int = 224,
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
        for param in self.final_layer.parameters():
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
