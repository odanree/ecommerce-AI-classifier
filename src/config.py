"""
Configuration file for E-commerce CLIP Fine-tuning Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset configuration
DATASET_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seed': 42,
    'image_size': 224,
    'image_extensions': ['.jpg', '.jpeg', '.png', '.webp']
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'ViT-B/32',  # Options: 'RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
    'freeze_backbone': False,
    'clip_dim': 512,  # ViT-B/32 and RN50: 512, ViT-L/14: 768
    'classifier_hidden_dims': [512, 256],
    'dropout_rates': [0.3, 0.2]
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_workers': 4,
    'save_every': 5,
    'use_mixed_precision': True,
    'gradient_clip': 1.0
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    'type': 'AdamW',
    'lr': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,
    'eps': 1e-8
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    'type': 'CosineAnnealingLR',
    'T_max': 10,
    'eta_min': 1e-6
}

# Evaluation configuration
EVAL_CONFIG = {
    'batch_size': 32,
    'top_k': 5,
    'visualize_samples': 16,
    'save_confusion_matrix': True,
    'save_per_class_metrics': True,
    'save_sample_predictions': True,
    'compare_zero_shot': True
}

# Demo configuration
DEMO_CONFIG = {
    'top_k': 5,
    'confidence_threshold': 0.5,
    'save_visualizations': True
}

# CLIP normalization parameters (from OpenAI CLIP)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Logging configuration
LOGGING_CONFIG = {
    'log_file': 'training.log',
    'log_level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Device configuration
def get_device():
    """Get the best available device"""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

DEVICE = get_device()

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        NOTEBOOKS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Created all project directories")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
