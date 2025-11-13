"""
Unit tests for train_clip.py module.
Tests CLIP model initialization, forward pass, and training utilities.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_clip import CLIPClassifier, train_epoch, validate
from dataset_prep import EcommerceDataset, get_default_transforms
from PIL import Image


@pytest.fixture
def sample_model():
    """Create a sample CLIPClassifier model."""
    return CLIPClassifier(
        num_classes=4,
        model_name="ViT-B/32",
        freeze_backbone=False
    )


@pytest.fixture
def sample_dataset_dir():
    """Create temporary dataset with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        categories = ["shoes", "bags", "shirts", "electronics"]
        
        for category in categories:
            category_dir = tmpdir_path / category
            category_dir.mkdir()
            for i in range(5):
                img = Image.new('RGB', (100, 100), color='white')
                img.save(category_dir / f"{category}_{i}.png")
        
        yield tmpdir_path


class TestCLIPClassifier:
    """Test suite for CLIPClassifier model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = CLIPClassifier(num_classes=4)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_model_has_required_attributes(self):
        """Test that model has required attributes."""
        model = CLIPClassifier(num_classes=4)
        assert hasattr(model, 'clip_model')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'model_name')
    
    def test_model_different_class_counts(self):
        """Test model with different number of classes."""
        for num_classes in [2, 4, 10, 20]:
            model = CLIPClassifier(num_classes=num_classes)
            assert model is not None
    
    def test_model_different_architectures(self):
        """Test model with different CLIP architectures."""
        # Note: This may require downloading models, so we test with ViT-B/32
        model = CLIPClassifier(num_classes=4, model_name="ViT-B/32")
        assert model.model_name == "ViT-B/32"
    
    def test_model_freeze_backbone(self):
        """Test model with frozen backbone."""
        model_frozen = CLIPClassifier(num_classes=4, freeze_backbone=True)
        model_unfrozen = CLIPClassifier(num_classes=4, freeze_backbone=False)
        
        # Check that frozen model has fewer trainable parameters
        frozen_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        unfrozen_params = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
        
        assert frozen_params < unfrozen_params
    
    def test_forward_pass_single_image(self, sample_model):
        """Test forward pass with single image."""
        batch = torch.randn(1, 3, 224, 224)
        output = sample_model(batch)
        
        assert output.shape == (1, 4)  # batch_size=1, num_classes=4
    
    def test_forward_pass_batch(self, sample_model):
        """Test forward pass with batch of images."""
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            batch = torch.randn(batch_size, 3, 224, 224)
            output = sample_model(batch)
            assert output.shape == (batch_size, 4)
    
    def test_forward_pass_output_type(self, sample_model):
        """Test that forward pass output is correct type."""
        batch = torch.randn(4, 3, 224, 224)
        output = sample_model(batch)
        assert isinstance(output, torch.Tensor)
    
    def test_model_device_transfer(self, sample_model):
        """Test moving model to different devices."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_on_device = sample_model.to(device)
        
        # Check that parameters are on correct device
        for param in model_on_device.parameters():
            assert param.device.type == device.split(':')[0]
    
    def test_classifier_is_sequential(self, sample_model):
        """Test that classifier is properly structured."""
        assert isinstance(sample_model.classifier, nn.Sequential)
    
    def test_model_eval_mode(self, sample_model):
        """Test model can be set to eval mode."""
        sample_model.eval()
        assert not sample_model.training
    
    def test_model_train_mode(self, sample_model):
        """Test model can be set to train mode."""
        sample_model.train()
        assert sample_model.training


class TestTrainingUtilities:
    """Test suite for training functions."""
    
    def test_train_epoch_basic(self, sample_model, sample_dataset_dir):
        """Test basic train_epoch execution."""
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=get_default_transforms())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=1e-5)
        device = "cpu"
        
        # Run one epoch
        result = train_epoch(
            model=sample_model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # train_epoch returns (loss, accuracy)
        assert isinstance(result, tuple)
        loss, accuracy = result
        assert isinstance(loss, (float, int))
        assert isinstance(accuracy, (float, int))
        assert loss >= 0
        assert 0 <= accuracy <= 100
    
    def test_validate_basic(self, sample_model, sample_dataset_dir):
        """Test basic validate execution."""
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=get_default_transforms())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        criterion = nn.CrossEntropyLoss()
        device = "cpu"
        sample_model.eval()
        
        # Run validation
        val_loss, val_acc = validate(
            model=sample_model,
            dataloader=dataloader,
            criterion=criterion,
            device=device
        )
        
        assert isinstance(val_loss, (float, int))
        assert isinstance(val_acc, (float, int))
        assert val_loss >= 0
        assert 0 <= val_acc <= 100  # Accuracy is in percentage


class TestModelOutput:
    """Test suite for model output properties."""
    
    def test_output_is_logits(self, sample_model):
        """Test that model output represents logits."""
        batch = torch.randn(4, 3, 224, 224)
        output = sample_model(batch)
        
        # Logits should have reasonable range
        assert output.min() >= -1000
        assert output.max() <= 1000
    
    def test_output_can_be_probabilities(self, sample_model):
        """Test that logits can be converted to probabilities."""
        batch = torch.randn(4, 3, 224, 224)
        output = sample_model(batch)
        
        # Convert to probabilities
        probs = torch.softmax(output, dim=1)
        
        assert probs.shape == output.shape
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_output_top_k_predictions(self, sample_model):
        """Test getting top-k predictions."""
        batch = torch.randn(2, 3, 224, 224)
        output = sample_model(batch)
        probs = torch.softmax(output, dim=1)
        
        # Get top 2 predictions
        top_k = torch.topk(probs, 2, dim=1)
        
        assert top_k.values.shape == (2, 2)
        assert top_k.indices.shape == (2, 2)


class TestGradientFlow:
    """Test suite for gradient flow during training."""
    
    def test_gradients_computed(self, sample_model):
        """Test that gradients are computed."""
        batch = torch.randn(2, 3, 224, 224)
        output = sample_model(batch)
        loss = output.sum()
        
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = any(p.grad is not None for p in sample_model.parameters())
        assert has_gradients
    
    def test_gradients_are_non_zero(self, sample_model):
        """Test that computed gradients are non-zero."""
        batch = torch.randn(2, 3, 224, 224)
        output = sample_model(batch)
        loss = output.sum()
        
        loss.backward()
        
        # Check that gradients are computed and non-zero
        non_zero_grads = 0
        for p in sample_model.parameters():
            if p.grad is not None and (p.grad != 0).any():
                non_zero_grads += 1
        
        assert non_zero_grads > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
