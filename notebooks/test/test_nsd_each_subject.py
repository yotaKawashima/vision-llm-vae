"""Test NSDStimulusDataset with per-subject mode."""

import os
import sys
from unittest.mock import patch, MagicMock

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from experiments.neuralnet.datasets import NSDStimulusDataset


def test_nsd_per_subject_dataset():
    """Test NSDStimulusDataset with subject parameter."""

    num_total_images = 1000
    num_subject_images = 100
    img_height, img_width, channels = 425, 425, 3
    embedding_dim = 768

    # Mock data
    nsd_ids = np.arange(num_subject_images)
    dummy_stimulus_info = pd.DataFrame({"nsdId": nsd_ids})
    dummy_embeddings = torch.randn(num_total_images, embedding_dim)

    with patch("experiments.neuralnet.datasets.h5py.File") as mock_h5, \
         patch("experiments.neuralnet.datasets.pd.read_pickle") as mock_pickle, \
         patch("experiments.neuralnet.datasets.torch.load") as mock_torch_load:

        # Setup mocks
        mock_file = MagicMock()
        mock_images = np.random.randint(
            0, 255, (num_total_images, img_height, img_width, channels), dtype=np.uint8
        )
        mock_file.__getitem__.return_value = mock_images
        mock_h5.return_value.__enter__.return_value = mock_file
        mock_h5.return_value.__exit__.return_value = None

        mock_pickle.return_value = dummy_stimulus_info
        mock_torch_load.return_value = dummy_embeddings

        # Initialize dataset with subject
        dataset = NSDStimulusDataset(
            subject=1,
            nsd_stimulus_info_this_subject_path="dummy_path.pkl",
            nsd_stimulus_path="dummy_path.hdf5",
            nsd_text_embeddings_path="dummy_path.pt",
        )

        # Tests
        assert len(dataset) == num_subject_images
        print(f"✓ Dataset length: {len(dataset)}")

        # Test __getitem__
        sample = dataset[0]
        assert "image" in sample
        assert "text_embedding" in sample
        assert "nsd_id" in sample
        print(f"✓ Sample keys: {list(sample.keys())}")

        assert isinstance(sample["image"], torch.Tensor)
        print(f"✓ Image shape: {sample['image'].shape}")

        assert isinstance(sample["text_embedding"], torch.Tensor)
        print(f"✓ Text embedding shape: {sample['text_embedding'].shape}")

        assert sample["nsd_id"] == 0
        print(f"✓ NSD ID: {sample['nsd_id']}")

        # Verify context manager was called (file properly closed)
        mock_h5.return_value.__enter__.assert_called()
        mock_h5.return_value.__exit__.assert_called()
        print("✓ HDF5 file properly closed (with statement)")

        print("\n✅ All per-subject tests passed!")


if __name__ == "__main__":
    test_nsd_per_subject_dataset()
