"""Test NSDStimulusDataset with mocks."""

import os
import sys
from unittest.mock import patch, MagicMock

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from experiments.neuralnet.datasets import NSDStimulusDataset


def test_nsd_stimulus_dataset_initialization():
    """Test NSDStimulusDataset initialization with mocked dependencies."""

    # Create dummy data
    num_samples = 100
    img_height, img_width, channels = 425, 425, 3
    embedding_dim = 768

    # Dummy stimulus info
    dummy_stimulus_info = pd.DataFrame({
        'cocoId': list(range(num_samples))
    })

    # Dummy text embeddings
    dummy_embeddings = torch.randn(num_samples, embedding_dim)

    # Mock h5py.File
    with patch('experiments.neuralnet.datasets.h5py.File') as mock_h5, \
         patch('experiments.neuralnet.datasets.pd.read_pickle') as mock_pickle, \
         patch('experiments.neuralnet.datasets.torch.load') as mock_torch_load:

        # Setup h5py mock
        mock_file = MagicMock()
        mock_images = np.random.randint(0, 255, (num_samples, img_height, img_width, channels), dtype=np.uint8)
        mock_file.__getitem__.return_value = mock_images
        mock_h5.return_value = mock_file

        # Setup pd.read_pickle mock
        mock_pickle.return_value = dummy_stimulus_info

        # Setup torch.load mock
        mock_torch_load.return_value = dummy_embeddings

        # Initialize dataset
        dataset = NSDStimulusDataset(
            nsd_stimulus_info_path="dummy_path.pkl",
            nsd_stimulus_path="dummy_path.hdf5",
            nsd_text_embeddings_path="dummy_path.pt"
        )

        # Assertions
        assert len(dataset) == num_samples
        print(f"✓ Dataset length: {len(dataset)}")

        assert dataset.ids == list(range(num_samples))
        print("✓ Image IDs loaded correctly")

        assert dataset.text_embeddings.shape == (num_samples, embedding_dim)
        print(f"✓ Text embeddings shape: {dataset.text_embeddings.shape}")

        # Check that h5file was accessed correctly
        mock_h5.assert_called_once_with("dummy_path.hdf5", "r")
        print("✓ HDF5 file accessed correctly")


def test_nsd_stimulus_dataset_getitem():
    """Test NSDStimulusDataset __getitem__ with mocked dependencies."""

    num_samples = 100
    img_height, img_width, channels = 425, 425, 3
    embedding_dim = 768

    dummy_stimulus_info = pd.DataFrame({
        'cocoId': list(range(num_samples))
    })
    dummy_embeddings = torch.randn(num_samples, embedding_dim)

    with patch('experiments.neuralnet.datasets.h5py.File') as mock_h5, \
         patch('experiments.neuralnet.datasets.pd.read_pickle') as mock_pickle, \
         patch('experiments.neuralnet.datasets.torch.load') as mock_torch_load:

        # Setup mocks
        mock_file = MagicMock()
        mock_images = np.random.randint(0, 255, (num_samples, img_height, img_width, channels), dtype=np.uint8)
        mock_file.__getitem__.return_value = mock_images
        mock_h5.return_value.__enter__.return_value = mock_file
        mock_h5.return_value.__exit__.return_value = None

        mock_pickle.return_value = dummy_stimulus_info
        mock_torch_load.return_value = dummy_embeddings

        dataset = NSDStimulusDataset(
            nsd_stimulus_info_path="dummy_path.pkl",
            nsd_stimulus_path="dummy_path.hdf5",
            nsd_text_embeddings_path="dummy_path.pt"
        )

        # Test __getitem__
        sample = dataset[0]

        assert "image" in sample
        assert "text_embedding" in sample
        print(f"✓ Sample keys: {list(sample.keys())}")

        assert isinstance(sample["image"], torch.Tensor)
        print(f"✓ Image is torch.Tensor: {sample['image'].shape}")

        assert isinstance(sample["text_embedding"], torch.Tensor)
        print(f"✓ Text embedding is torch.Tensor: {sample['text_embedding'].shape}")

        print("✅ __getitem__ tests passed!")


if __name__ == "__main__":
    test_nsd_stimulus_dataset_initialization()
    test_nsd_stimulus_dataset_getitem()
