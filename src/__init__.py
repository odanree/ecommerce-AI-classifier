"""
E-commerce CLIP Classifier Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .dataset_prep import EcommerceDataset, create_dataloaders
from .train_clip import CLIPClassifier, train_clip
from .evaluate import evaluate_model, run_evaluation
from .demo_classify import classify_image

__all__ = [
    "EcommerceDataset",
    "create_dataloaders",
    "CLIPClassifier",
    "train_clip",
    "evaluate_model",
    "run_evaluation",
    "classify_image",
]
