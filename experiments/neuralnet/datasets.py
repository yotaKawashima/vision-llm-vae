"""CocoTextEmbeddingImageDataset: PyTorch Dataset for COCO images with text embeddings."""

import torch
import json
import os
import sys
from typing import Union

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd

from .logger import Logger


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import config


class ApplyTransformSubset(torch.utils.data.Dataset):
    """
    ApplyTransformSubset
    -----------------------------
    A PyTorch Dataset wrapper that applies a specified transform to the "image" field of samples from an underlying subset dataset.
    This class is designed to take an existing dataset (or subset) that returns samples containing a "image" key, and apply a given image transform to that raw image before returning the sample. The transformed image is returned under the "image" key in the output sample dictionary.
    Parameters
    ----------
    subset : torch.utils.data.Dataset
        The underlying dataset (or subset) that provides samples. Each sample from this dataset is expected
        to be a dictionary containing at least a "image" key with a PIL.Image or similar object.
    transform : callable
        A function or torchvision.transforms pipeline that takes a PIL.Image and returns a transformed version (e.g., a tensor). This transform will be applied to the "image" field of each sample from the subset.
    Methods    -------
    __getitem__(index)
        Retrieves a sample from the underlying subset at the specified index, applies the transform to the "image" field, and returns a new sample dictionary containing the transformed image under the "image"        key, along with any other fields from the original sample.
    __len__()
        Returns the length of the underlying subset dataset.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Get the sample from the underlying subset
        sample = self.subset[index]
        if self.transform is not None:
            # Apply the transform to the "image" field and store it in the "image" key
            sample["image"] = self.transform(sample["image"])
        return sample

    def __len__(self):
        return len(self.subset)


class CocoTextEmbeddingImageDataset(Dataset):
    """
    CocoTextEmbeddingImageDataset
    -----------------------------

    PyTorch Dataset that pairs COCO images with precomputed caption text embeddings.

    This dataset reads:
    - a COCO image directory (train or validation) determined by `split`,
    - a JSON metadata file containing image ids,
    - a serialized tensor/file of text embeddings (loaded with torch.load).

    Each item returned by __getitem__ is a dictionary with keys:
    - "image": the image (PIL.Image or transformed tensor),
    - "text_embedding": the corresponding text embedding (torch.Tensor or array-like),
    - "image_id": the integer COCO image id.

    Parameters
    ----------
    split : str
        Which split to use. Expected values typically are "train" or "val".
        The value determines which image directory and which metadata / embedding
        file paths are read from the global `config`.
    img_transform : callable, optional
        Optional transform to apply to PIL.Image objects (e.g., a torchvision
        transforms pipeline). If provided, it is applied to the image before it
        is returned.

    Attributes
    ----------
    split : str
        The dataset split being used ("train" or "val").
    img_dir : str
        Filesystem path to the directory containing COCO images for the selected
        split.
    text_embeddings : torch.Tensor or list
        Loaded text embeddings corresponding to entries in the metadata file.
        Indexing into this object by dataset index yields the embedding for the
        returned sample.
    image_ids : list of int
        List of COCO image ids (integers) read from the metadata JSON file. These
        are used to construct image filenames using zero-padded 12-digit format
        (e.g. "000000123456.jpg").
    img_transform : callable
        The image transform function or pipeline to apply to loaded images.
        By default, this is set to `config.img_transform`.

    Raises
    ------
    FileNotFoundError
        If the metadata file, the text embeddings file, or an image file cannot be
        found when accessed.
    ValueError, KeyError
        If the metadata JSON structure does not contain the expected "image_id"
        entries or if loaded objects have incompatible shapes.

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> dataset = CocoTextEmbeddingImageDataset("val", img_transform=my_transform)
    >>> loader = DataLoader(dataset, batch_size=8, shuffle=False)
    >>> image, text_embedding, ids = next(iter(loader))
    """

    def __init__(
        self,
        split,
        img_transform=None,
    ):
        self.split = split
        if not config.text_embeddings_summary:
            raise NotImplementedError(
                "Only text_embeddings_summary=True is supported now."
            )

        # image
        if split == "train":
            self.img_dir = config.coco_image_train_dir_path
            text_embedding_path = config.text_embeddings_train_path
            meta_data_path = config.text_embeddings_meta_train_path
        elif split == "val":
            self.img_dir = config.coco_image_val_dir_path
            text_embedding_path = config.text_embeddings_val_path
            meta_data_path = config.text_embeddings_meta_val_path
        else:
            raise ValueError(f"Invalid split value: {split}")

        # load image ids from meta data
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)
        self.image_ids = [int(item["image_id"]) for item in meta_data]
        del meta_data

        # load text embeddings
        self.text_embeddings = torch.load(
            text_embedding_path, map_location=torch.device("cpu")
        )

        self.img_transform = img_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.image_ids[idx]:012d}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            image = self.img_transform(image)
        text_embedding = self.text_embeddings[idx].squeeze(0)  # 1, D -> D

        return {
            "image": image,
            "text_embedding": text_embedding,
            "image_id": self.image_ids[idx],
        }


# Available embedding keys in the HDF5 file
H5_EMBEDDING_KEYS = [
    "all_mpnet_base_v2_mean_embeddings",  # (N, 768)
]


class CocoH5Dataset(Dataset):
    """
    CocoH5Dataset
    -------------
    PyTorch Dataset that reads COCO images and embeddings directly from an HDF5
    file (e.g. ms_coco_embeddings_square256_proper_chunks.h5).

    The HDF5 file must have the following structure::
        /train
            /data                              (N, 256, 256, 3)  uint8 HWC
            /coco_ids                          (N,)
            /all_mpnet_base_v2_mean_embeddings (N, 768)
            /image_size_hwc                    (N, 3)
        /val  (same structure)

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    split : str
        "train" or "val".
    embedding_key : str
        Which embedding dataset to load as ``text_embedding``.
        Default: ``"all_mpnet_base_v2_mean_embeddings"``.
    img_transform : callable, optional
        Transform applied to the image tensor.
        If ``None``, the raw uint8 HWC image from the HDF5 file is returned.
    Notes
    -----
    All images and embeddings are loaded eagerly into RAM during ``__init__``
    so that DataLoader multi-process workers can access data via forked memory
    without needing individual HDF5 file handles.

    Returns (per item)
    ------------------
    dict with keys:
        "image"           torch.Tensor (C, H, W) float
        "text_embedding"  torch.Tensor (D,) or (5, D) float
        "image_id"        int  (COCO image id)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        embedding_key: str = "all_mpnet_base_v2_mean_embeddings",
        img_transform=None,
        logger: Logger = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val' got '{split}'")
        if embedding_key not in H5_EMBEDDING_KEYS:
            raise ValueError(
                f"embedding_key must be one of {H5_EMBEDDING_KEYS}, got '{embedding_key}'"
            )

        self.h5_path = h5_path
        self.split = split
        self.embedding_key = embedding_key
        self.img_transform = img_transform

        if logger is not None:
            logger.log_info(
                f"Loading {split} dataset into RAM. This might take a minute..."
            )
        self.h5_file = h5py.File(h5_path, "r")
        self.coco_ids = self.h5_file[split]["coco_ids"][:].tolist()
        self._len = len(self.coco_ids)  # number of samples in this split
        self.images_np = self.h5_file[split]["data"]
        self.embeddings_np = self.h5_file[split][self.embedding_key]

        if logger is not None:
            logger.log_info("Done loading into RAM!")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Image: uint8 HWC
        img_np = self.images_np[idx]
        if self.img_transform is not None:
            image = self.img_transform(img_np)
        else:
            image = img_np

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Embedding
        emb_np = self.embeddings_np[idx]
        # make sure that embeddings are L2 normalized (they should already be, but just in case)
        norms = np.linalg.norm(emb_np)
        emb_np = emb_np / (norms + 1e-12)
        text_embedding = torch.from_numpy(emb_np)

        return {
            "image": image,
            "text_embedding": text_embedding,
            "image_id": self.coco_ids[idx],
        }

    def __del__(self):
        self.h5_file.close()


class NSDStimulusDataset(Dataset):
    """
    NSDStimulusDataset
    -----------------------------
    PyTorch Dataset that provides images and text embeddings from the NSD Stimulus dataset.

    Parameters
    ----------
    subject : int, str, or None, optional
        Subject identifier. If None (default), loads all stimuli.
        If "special515", uses special515 dataset. Otherwise, should be an integer corresponding
        to a subject.
    nsd_stimulus_info_path : str, optional
        Path to the NSD stimulus info pickle file. (nsd_stim_info_merged.pkl)
        Used when subject is None.
    nsd_stimulus_path : str, optional
        Path to the NSD stimulus HDF5 file. (nsd_stimuli.hdf5)
    nsd_text_embeddings_path : str, optional
        Path to the precomputed text embeddings for NSD stimuli. (pt file)
    img_transform : callable, optional
        A function or torchvision.transforms pipeline to apply to the images. If None, no transform is applied.
    """

    def __init__(
        self,
        subject: Union[int, str, None] = None,
        nsd_stimulus_info_path=config.nsd_stimulus_info_path,
        nsd_stimulus_path=config.nsd_stimulus_path,
        nsd_text_embeddings_path=config.text_embeddings_nsd_path,
        img_transform=None,
    ):
        self.subject = subject
        self.img_transform = img_transform
        self._is_per_subject = subject is not None

        # load stimulus info
        if self._is_per_subject:
            if nsd_stimulus_info_path == config.nsd_stimulus_info_path:
                raise ValueError(
                    "When subject is specified, nsd_stimulus_info_path must be provided and should be different from the default config.nsd_stimulus_info_path"
                )
            self.stimulus_info = pd.read_pickle(nsd_stimulus_info_path)
            self.ids = self.stimulus_info["nsdId"].tolist()
            self.id_key = "nsd_id"
        else:
            self.stimulus_info = pd.read_pickle(nsd_stimulus_info_path)
            self.ids = self.stimulus_info["cocoId"].tolist()
            self.id_key = "image_id"

        self._len = len(self.ids)

        # load image stimuli in NSD
        # Note that we read all stimuli into RAM here.
        with h5py.File(nsd_stimulus_path, "r") as h5_file:
            images_np = h5_file["imgBrick"][:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load text embeddings
        text_embeddings = torch.load(
            nsd_text_embeddings_path, map_location=torch.device(device)
        )
        # make sure that embeddings are L2 normalized (they should already be, but just in case)
        norms = torch.linalg.norm(text_embeddings, dim=1, keepdim=True)
        text_embeddings = text_embeddings / (norms + 1e-12)

        # extract the stimuli corresponding to this subject if per-subject
        if self._is_per_subject:
            self.images_np = images_np[self.ids, :, :, :]
            self.text_embeddings = text_embeddings[self.ids, :]
        else:
            self.images_np = images_np
            self.text_embeddings = text_embeddings

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Image: uint8 HWC
        img_np = self.images_np[idx]
        if self.img_transform is not None:
            image = self.img_transform(img_np)
        else:
            image = img_np

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Embedding
        text_embedding = self.text_embeddings[idx]

        return {
            "image": image,
            "text_embedding": text_embedding,
            self.id_key: self.ids[idx],
        }
