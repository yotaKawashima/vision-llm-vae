"""Module providing functions for extracting model activations."""

import sys
import math
from pathlib import Path
from typing import List, Dict, Optional, Union


import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.neuralnet.logger import Logger


class ActivationExtractor:
    """
    Extracts activations from a model during inference.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        logger: Logger,
        target_layers: Union[str, List[str]],
    ):
        """
        Initializes the ActivationExtractor.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model from which to extract activations.
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the data to pass through the model for activation extraction.
        logger : Logger
            Logger instance for logging messages.
        target_layers : Union[str, List[str]]
            Layer name(s) to extract activations from. Can be a single string for one layer

        """
        self.model = model
        self.model.eval()
        self.dataloader = dataloader
        self.logger = logger
        self.target_layers = target_layers
        self.hook_handles: List = []
        # store per-layer activations
        self.total_samples = len(self.dataloader.dataset)
        self.activations: Dict[str, torch.Tensor] = {}
        self._activations_idx: Dict[str, int] = {}

        # Register forward hooks on target layers
        self.register_hooks()

    def register_hooks(self):
        """
        Register forward hooks on target layers.
        """

        for name, module in self.model.named_modules():
            if self._match_layer_name(name):
                handle = module.register_forward_hook(self._get_hook(name))
                self.hook_handles.append(handle)

        if len(self.hook_handles) == 0:
            self.logger.log_error("No layers were matched for hooking.")
            sys.exit(1)

        self.logger.log_success(f"Registered {len(self.hook_handles)} hooks")

    def _match_layer_name(self, layer_name: str):
        """
        Check if a layer name matches target_layers
        Parameters
        ----------
        layer_name : str
            The name of the layer to check.

        Returns
        -------
        bool
            True if the layer matches target layers, False otherwise.
        """
        # If target_layers is a list, check for exact or partial matches
        if isinstance(self.target_layers, list):
            # if "all" is in target_layers
            if "all" in self.target_layers:
                return True

            # Exact match (e.g. target_layers=['layer1.1.conv2'])
            if layer_name in self.target_layers:
                return True
            # Partial match (e.g. target_layers=['conv2', 'relu'])
            if any(target in layer_name for target in self.target_layers):
                return True

        # If target_layers is a string, check for exact or partial match
        elif isinstance(self.target_layers, str):
            if self.target_layers == "all":
                return True
            # Exact match or Partial match
            if layer_name == self.target_layers or self.target_layers in layer_name:
                return True

        else:
            raise ValueError(
                "Invalid type for target_layers. Expected str or List[str]."
            )

        return False

    def _get_hook(self, layer_name: str):
        """
        Create a forward hook for capturing activations.
        Keep the activations in preallocated tensors.

        Parameters
        ----------
        layer_name : str
            Layer name for identification

        Returns
        -------
        callable
            Hook function to be registered with PyTorch module
        """

        def hook(_module, _input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store full activation for current batch.
            # output shape: (batch, )
            batch_act = output.detach().cpu()
            # dtype = float16 to save memory
            batch_act = batch_act.to(torch.float16)

            self._store_activations(layer_name, batch_act)

        return hook

    def _store_activations(self, layer_name: str, activations: torch.Tensor):
        """
        Store activations in preallocated tensors.

        Parameters
        ----------
        layer_name : str
            Layer name for identification
        activations : torch.Tensor
            Activations to be stored, shape (batch_size, ...)
        """

        batch_size = activations.size(0)

        # first time seeing this layer
        if layer_name not in self.activations:
            # create preallocated tensor on CPU
            shape_rest = activations.shape[1:]
            flat_dim = math.prod(shape_rest)
            self.activations[layer_name] = torch.empty(
                (self.total_samples, flat_dim), dtype=activations.dtype
            )
            self._activations_idx[layer_name] = 0

        # store data in the corresponding location in self.activations
        start_idx = self._activations_idx[layer_name]
        end_idx = start_idx + batch_size
        # store after flattening it.
        self.activations[layer_name][start_idx:end_idx].copy_(
            activations.reshape(activations.shape[0], -1)
        )
        self._activations_idx[layer_name] = end_idx

        return

    def extract(
        self,
        input_modality: str = "image",
        vision_bias: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from the model using the provided dataloader.

        Parameters
        ----------
        input_modality : str, optional
            The mode of extraction, either "image", "text", or "both". Defaults to "image".
        vision_bias: float, optional
            Bias toward vision when combining text embedings and visual latent values (only for "both" mode)

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary mapping layer names to their corresponding activations. Each value is a tensor of shape.
        """
        # Checks if CUDA is available
        is_cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if is_cuda_available else "cpu")

        self.model.to(device)

        self.logger.log_info("Starting extracting activations ...")

        with torch.inference_mode():
            for batch in self.dataloader:
                # Gets the current batch
                images = batch["image"].to(device)
                text_embeddings = batch["text_embedding"].to(device)

                # Forward pass through the model
                self._forward(
                    input_modality,
                    images=images,
                    text_embeddings=text_embeddings,
                    vision_bias=vision_bias,
                )

        # Reports that the extraction has finished
        self.logger.log_success("Finished extraction")

        return self.activations

    def _forward(
        self,
        input_modality: str,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        vision_bias: Optional[float] = None,
    ):
        """
        Forward pass to extract activations and latent values.
        Registered hooks automatically keep activations but not latent values.
        This code manually stores latent values.

        Parameters
        ----------
        input_modality : str
            The mode of extraction, either "image", "text", or "both". Defaults to "image".
        images: torch.Tensor
            batch images
        text_embeddings: torch.Tensor
            batch text embeddings
        vision_bias: Optional[float]
            bias toward vision (weight when combining text embedings and visual latent values)

        """
        # For "image" modality, we will extract activations by passing images through the model as usual.
        if input_modality == "image":
            encoded = self.model.encode(images)

            # Case: VAE (returns mu and log_var)
            if isinstance(encoded, tuple):
                mu, log_var = encoded
                latent_vars = self.model.reparameterize(mu, log_var)
                # manually store latent_vars since hooks won't capture them
                self._store_activations("mu", mu.detach().cpu().to(torch.float16))
                self._store_activations(
                    "latent", latent_vars.detach().cpu().to(torch.float16)
                )

            # Case: AE (returns latent directly)
            else:
                latent_vars = encoded
                # manually store latent_vars since hooks won't capture them
                self._store_activations(
                    "latent", latent_vars.detach().cpu().to(torch.float16)
                )
            # Decoding (to trigger hooks in the decoder layers)
            _ = self.model.decode(latent_vars)

        else:
            # For "text" and "both" modalities, we will handle the encoding differently below.
            raise NotImplementedError(
                f"Encoding for input_modality '{input_modality}' is not implemented."
            )

    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup()
