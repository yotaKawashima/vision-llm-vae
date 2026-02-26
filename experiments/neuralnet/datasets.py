"""CocoTextEmbeddingImageDataset: PyTorch Dataset for COCO images with text embeddings."""

import torch
import json
import os
import sys
import random
from typing import Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd

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
        img_transform=config.img_transform,
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


class NSDStimulusDataset(Dataset):
    """
    NSDStimulusDataset
    -----------------------------

    PyTorch Dataset that provides images from the NSD Stimulus dataset.

    """

    def __init__(
        self,
        subject: Union[int, str],
        nsd_stimulus_info_path=config.nsd_stimulus_info_path,
        img_transform=config.img_transform,
    ):

        # subject
        if isinstance(subject, int):
            if subject not in config.subjects:
                raise ValueError("Invalid subject")
            else:
                self.subject = f"subject{subject}_train"
        else:
            if subject == "test":
                self.subject = "test"
            else:
                raise ValueError("Invalid subject")

        # image
        self.coco_train_img_dir = config.coco_image_train_dir_path
        self.coco_val_img_dir = config.coco_image_val_dir_path

        # load stimulus info
        self.stimulus_info = pd.read_pickle(nsd_stimulus_info_path)
        self.stimulus_info_shared1000 = self.stimulus_info[
            self.stimulus_info.shared1000
        ]
        self.stimulus_info_subject = self._get_stimulus_info_subject(subject)
        self.img_transform = img_transform

        # load text embeddings
        text_embedding_train_path = config.text_embeddings_train_path
        text_embedding_test_path = config.text_embeddings_val_path
        self.text_embeddings_train = torch.load(
            text_embedding_train_path, map_location=torch.device("cpu")
        )
        self.text_embeddings_test = torch.load(
            text_embedding_test_path, map_location=torch.device("cpu")
        )

    def __len__(self):
        return len(self.stimulus_info_subject)

    def _get_stimulus_info_subject(self, subject: Union[str, int]):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError(
            "id will be different between text_embeddings and stimulus_info_subject, need to match them by image_id"
        )
        # image file path
        # if self.stimulus_info_subject.iloc[idx]["cocoSplit"] == "train2017":
        #     img_dir = self.coco_train_img_dir
        #     text_embeddings = self.text_embeddings_train
        # elif self.stimulus_info_subject.iloc[idx]["cocoSplit"] == "val2017":
        #     img_dir = self.coco_val_img_dir
        #     text_embeddings = self.text_embeddings_test
        # else:
        #     raise ValueError(
        #         f"Invalid cocoSplit value: {self.stimulus_info_subject.iloc[idx]['cocoSplit']}"
        #     )

        # image_id = self.stimulus_info_subject.iloc[idx]["cocoId"]
        # img_path = os.path.join(img_dir, f"{image_id:012d}.jpg")

        # image = Image.open(img_path).convert("RGB")
        # image = self.img_transform(image)

        # # idx will be different between text_embeddings and stimulus_info_subject.
        # # text_embedding = text_embeddings[idx].squeeze(0)  # 1, D -> D

        # return {
        #     "image": image,
        #     "text_embedding": text_embeddings,
        #     "image_id": image_id,
        # }
