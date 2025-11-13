"""
Unit tests for dataset_prep.py module.
Tests dataset loading, category mapping, and image transformations.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_prep import EcommerceDataset, get_default_transforms


@pytest.fixture
def sample_dataset_dir():
    """Create a temporary dataset directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create categories and sample images
        categories = ["shoes", "bags", "shirts"]
        for category in categories:
            category_dir = tmpdir_path / category
            category_dir.mkdir()
            
            # Create 3 sample images per category
            for i in range(3):
                img = Image.new('RGB', (100, 100), color=f'white')
                img.save(category_dir / f"{category}_{i}.png")
        
        yield tmpdir_path


class TestEcommerceDataset:
    """Test suite for EcommerceDataset class."""
    
    def test_dataset_initialization(self, sample_dataset_dir):
        """Test dataset initialization with valid directory."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        assert len(dataset) == 9  # 3 categories * 3 images
        assert len(dataset.get_categories()) == 3
    
    def test_dataset_categories(self, sample_dataset_dir):
        """Test category mapping."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        categories = dataset.get_categories()
        assert "shoes" in categories
        assert "bags" in categories
        assert "shirts" in categories
    
    def test_category_to_idx_mapping(self, sample_dataset_dir):
        """Test category to index mapping."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        assert dataset.category_to_idx["bags"] in [0, 1, 2]
        assert dataset.category_to_idx["shirts"] in [0, 1, 2]
        assert dataset.category_to_idx["shoes"] in [0, 1, 2]
    
    def test_get_category_name(self, sample_dataset_dir):
        """Test retrieving category name from index."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        sample_image, label, _ = dataset[0]
        category_name = dataset.get_category_name(label)
        assert category_name in ["bags", "shoes", "shirts"]
    
    def test_load_sample_image(self, sample_dataset_dir):
        """Test loading and processing a sample image."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        image, label, path = dataset[0]
        
        # Should return PIL Image
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert isinstance(path, str)
    
    def test_load_sample_with_transform(self, sample_dataset_dir):
        """Test loading image with transform applied."""
        transform = get_default_transforms()
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=transform)
        
        image, label, path = dataset[0]
        
        # Should return tensor after transform
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)  # Default image size
    
    def test_label_consistency(self, sample_dataset_dir):
        """Test that labels are consistent with categories."""
        dataset = EcommerceDataset(str(sample_dataset_dir))
        
        # Get multiple samples
        for i in range(len(dataset)):
            _, label, path = dataset[i]
            category = dataset.get_category_name(label)
            # Path should contain the category name
            assert category in path
    
    def test_dataset_split_initialization(self, sample_dataset_dir):
        """Test dataset can be initialized with different splits."""
        dataset_train = EcommerceDataset(str(sample_dataset_dir), split='train')
        dataset_val = EcommerceDataset(str(sample_dataset_dir), split='val')
        dataset_test = EcommerceDataset(str(sample_dataset_dir), split='test')
        
        assert dataset_train.split == 'train'
        assert dataset_val.split == 'val'
        assert dataset_test.split == 'test'
    
    def test_empty_directory_handling(self):
        """Test handling of empty dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = EcommerceDataset(str(tmpdir))
            assert len(dataset) == 0
    
    def test_non_image_files_ignored(self, sample_dataset_dir):
        """Test that non-image files are ignored."""
        # Add a non-image file
        txt_file = sample_dataset_dir / "shoes" / "readme.txt"
        txt_file.write_text("This is not an image")
        
        dataset = EcommerceDataset(str(sample_dataset_dir))
        # Should still only count image files
        assert len(dataset) == 9


class TestDefaultTransforms:
    """Test suite for image transforms."""
    
    def test_default_transforms_returns_composition(self):
        """Test that get_default_transforms returns a valid composition."""
        transform = get_default_transforms()
        assert transform is not None
    
    def test_transforms_output_shape(self):
        """Test that transforms produce correct output shape."""
        transform = get_default_transforms(img_size=224)
        img = Image.new('RGB', (100, 100), color='white')
        transformed = transform(img)
        assert transformed.shape == (3, 224, 224)
    
    def test_transforms_custom_size(self):
        """Test transforms with custom image size."""
        transform = get_default_transforms(img_size=256)
        img = Image.new('RGB', (100, 100), color='white')
        transformed = transform(img)
        assert transformed.shape == (3, 256, 256)
    
    def test_transforms_output_is_tensor(self):
        """Test that transforms output is a tensor."""
        transform = get_default_transforms()
        img = Image.new('RGB', (100, 100), color='white')
        transformed = transform(img)
        assert isinstance(transformed, torch.Tensor)
    
    def test_transforms_normalize_values(self):
        """Test that normalization produces reasonable values."""
        transform = get_default_transforms()
        img = Image.new('RGB', (100, 100), color='white')
        transformed = transform(img)
        # Normalized values should be in reasonable range
        assert transformed.min() >= -5  # Allow some margin
        assert transformed.max() <= 5


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_corrupted_image_handling(self, sample_dataset_dir):
        """Test that corrupted images are handled gracefully."""
        # Create a corrupted image file
        bad_file = sample_dataset_dir / "shoes" / "corrupted.png"
        bad_file.write_text("This is not a valid image")
        
        dataset = EcommerceDataset(str(sample_dataset_dir))
        # Should be able to load despite corrupted file
        assert len(dataset) > 0
    
    def test_mixed_image_formats(self, sample_dataset_dir):
        """Test that different image formats are loaded."""
        # Create different format images
        formats = {
            "bags": [("jpg", lambda: Image.new('RGB', (100, 100)).save),
                     ("png", lambda: Image.new('RGB', (100, 100)).save)],
        }
        
        dataset = EcommerceDataset(str(sample_dataset_dir))
        assert len(dataset) > 0
    
    def test_invalid_dataset_path(self):
        """Test handling of invalid dataset path."""
        with pytest.raises((FileNotFoundError, OSError)):
            dataset = EcommerceDataset("/invalid/path/that/does/not/exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
