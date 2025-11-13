"""
Unit tests for evaluate.py module.
Tests evaluation metrics, confusion matrix, and visualization.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import evaluate_model, compute_metrics, plot_confusion_matrix
from dataset_prep import EcommerceDataset, get_default_transforms
from train_clip import CLIPClassifier
from PIL import Image


@pytest.fixture
def sample_dataset_dir():
    """Create temporary dataset with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        categories = ["shoes", "bags", "shirts", "electronics"]
        
        for category in categories:
            category_dir = tmpdir_path / category
            category_dir.mkdir()
            for i in range(10):
                img = Image.new('RGB', (100, 100), color='white')
                img.save(category_dir / f"{category}_{i}.png")
        
        yield tmpdir_path


@pytest.fixture
def categories():
    """Return list of categories."""
    return ["shoes", "bags", "shirts", "electronics"]


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = CLIPClassifier(num_classes=4, model_name="ViT-B/32")
    return model


@pytest.fixture
def sample_predictions():
    """Create sample predictions and targets."""
    # Create realistic predictions and targets
    num_samples = 100
    num_classes = 4
    
    # Random predictions
    predictions = torch.randint(0, num_classes, (num_samples,))
    targets = torch.randint(0, num_classes, (num_samples,))
    
    return predictions, targets


class TestComputeMetrics:
    """Test suite for compute_metrics function."""
    
    def test_accuracy_perfect_predictions(self, categories):
        """Test accuracy with perfect predictions."""
        predictions = np.array([0, 1, 2, 3, 0, 1])
        labels = np.array([0, 1, 2, 3, 0, 1])
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert metrics['accuracy'] == 1.0
    
    def test_accuracy_random_predictions(self, sample_predictions, categories):
        """Test accuracy with random predictions."""
        predictions, labels = sample_predictions
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_accuracy_zero_predictions(self, categories):
        """Test accuracy with all wrong predictions."""
        predictions = np.array([0, 0, 0, 0])
        labels = np.array([1, 2, 3, 1])
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert metrics['accuracy'] == 0.0
    
    def test_metrics_dict_keys(self, sample_predictions, categories):
        """Test that metrics dict contains expected keys."""
        predictions, labels = sample_predictions
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        metrics = compute_metrics(predictions, labels, categories)
        
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'per_class']
        for key in expected_keys:
            assert key in metrics
    
    def test_precision_recall_range(self, sample_predictions, categories):
        """Test that precision and recall are in valid range."""
        predictions, labels = sample_predictions
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        metrics = compute_metrics(predictions, labels, categories)
        
        # Check valid ranges
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_f1_score_range(self, sample_predictions, categories):
        """Test that F1 score is in valid range."""
        predictions, labels = sample_predictions
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_per_class_metrics(self, sample_predictions, categories):
        """Test per-class metrics structure."""
        predictions, labels = sample_predictions
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        metrics = compute_metrics(predictions, labels, categories)
        
        per_class = metrics['per_class']
        assert len(per_class) == len(categories)
        
        for category in categories:
            assert category in per_class
            class_metrics = per_class[category]
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1_score' in class_metrics
            assert 'support' in class_metrics
    
    def test_accuracy_values(self, categories):
        """Test various accuracy values."""
        # Use all categories to avoid index errors
        predictions = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.array([0, 1, 1, 0, 2, 3, 3, 2])
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert metrics['accuracy'] == 0.5


class TestConfusionMatrix:
    """Test suite for confusion matrix operations."""
    
    def test_confusion_matrix_creation(self):
        """Test creating confusion matrix from predictions and labels."""
        from sklearn.metrics import confusion_matrix
        
        predictions = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        
        cm = confusion_matrix(labels, predictions)
        
        # For perfect predictions, diagonal should be non-zero
        assert cm.diagonal().sum() == 8
    
    def test_confusion_matrix_diagonal_perfect(self):
        """Test confusion matrix with perfect predictions."""
        from sklearn.metrics import confusion_matrix
        
        predictions = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        
        cm = confusion_matrix(labels, predictions)
        
        # Perfect predictions: diagonal contains 1s
        assert np.allclose(cm.diagonal(), np.array([1., 1., 1., 1.]))
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix_creates_figure(self, mock_savefig):
        """Test that plot_confusion_matrix creates figure."""
        cm = np.array([[10, 2], [1, 15]])
        categories_test = ["class_0", "class_1"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cm.png"
            plot_confusion_matrix(cm, categories_test, str(save_path))
            
            # Check that savefig was called
            mock_savefig.assert_called()
    
    def test_confusion_matrix_sum(self):
        """Test that confusion matrix sum equals total samples."""
        from sklearn.metrics import confusion_matrix
        
        predictions = np.random.randint(0, 4, 100)
        labels = np.random.randint(0, 4, 100)
        
        cm = confusion_matrix(labels, predictions)
        
        assert cm.sum() == 100


class TestEvaluateModel:
    """Test suite for full evaluate_model function."""
    
    def test_evaluate_model_basic(self, sample_model, sample_dataset_dir, categories):
        """Test basic model evaluation."""
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=get_default_transforms())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = "cpu"
        sample_model.eval()
        
        results = evaluate_model(
            model=sample_model,
            dataloader=dataloader,
            device=device,
            categories=categories
        )
        
        assert isinstance(results, dict)
    
    def test_evaluate_model_returns_predictions(self, sample_model, sample_dataset_dir, categories):
        """Test that evaluate_model returns predictions."""
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=get_default_transforms())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = "cpu"
        sample_model.eval()
        
        results = evaluate_model(
            model=sample_model,
            dataloader=dataloader,
            device=device,
            categories=categories
        )
        
        assert 'predictions' in results
        assert 'labels' in results
        assert 'paths' in results
        assert 'probabilities' in results
    
    def test_evaluate_model_predictions_shape(self, sample_model, sample_dataset_dir, categories):
        """Test that predictions have correct shape."""
        dataset = EcommerceDataset(str(sample_dataset_dir), transform=get_default_transforms())
        num_samples = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        device = "cpu"
        sample_model.eval()
        
        results = evaluate_model(
            model=sample_model,
            dataloader=dataloader,
            device=device,
            categories=categories
        )
        
        assert results['predictions'].shape[0] == num_samples
        assert results['labels'].shape[0] == num_samples
        assert results['probabilities'].shape == (num_samples, len(categories))


class TestMetricsEdgeCases:
    """Test edge cases in metrics computation."""
    
    def test_single_class_predictions(self, categories):
        """Test metrics when model predicts only one class."""
        predictions = np.array([0, 0, 0, 0])
        labels = np.array([0, 1, 2, 3])
        
        metrics = compute_metrics(predictions, labels, categories)
        
        # Should still compute metrics
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.25
    
    def test_per_class_support(self, categories):
        """Test that per-class support values are correct."""
        # Use all categories to avoid index errors
        predictions = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        labels = np.array([0, 1, 1, 0, 2, 3, 3, 2])
        
        metrics = compute_metrics(predictions, labels, categories)
        
        # Check support values add up to total samples
        total_support = sum(metrics['per_class'][cat]['support'] for cat in categories if cat in metrics['per_class'])
        assert total_support == 8
    
    def test_large_number_of_samples(self, categories):
        """Test metrics with many samples."""
        num_samples = 1000
        predictions = np.random.randint(0, 4, num_samples)
        labels = np.random.randint(0, 4, num_samples)
        
        metrics = compute_metrics(predictions, labels, categories)
        
        assert 'accuracy' in metrics
        assert 'per_class' in metrics


class TestVisualization:
    """Test visualization functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix_display(self, mock_show):
        """Test confusion matrix plotting."""
        cm = np.array([[10, 2], [1, 15]])
        categories = ["shoes", "bags"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cm.png"
            # This should not raise
            try:
                plot_confusion_matrix(cm, categories, str(save_path))
            except Exception as e:
                pytest.fail(f"plot_confusion_matrix raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
