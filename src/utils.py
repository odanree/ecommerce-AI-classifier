"""
Utility functions for E-commerce CLIP Fine-tuning Project
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_acc,
    val_loss,
    categories,
    model_name,
    save_path
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        val_loss: Validation loss
        categories: List of categories
        model_name: Name of CLIP model
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'num_classes': len(categories),
        'categories': categories,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    return checkpoint


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        checkpoint dict, model, optimizer
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint, model, optimizer


def save_json(data, filepath):
    """Save data as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to: {filepath}")


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    epochs = history['epochs']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    
    plt.show()


def get_class_distribution(dataset):
    """
    Get class distribution in dataset
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Dictionary with class counts
    """
    from collections import Counter
    
    label_counts = Counter(dataset.labels)
    categories = dataset.get_categories()
    
    distribution = {
        categories[label]: count 
        for label, count in sorted(label_counts.items())
    }
    
    return distribution


def plot_class_distribution(distribution, save_path=None):
    """
    Plot class distribution
    
    Args:
        distribution: Dictionary of class counts
        save_path: Path to save plot
    """
    categories = list(distribution.keys())
    counts = list(distribution.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, counts, alpha=0.7, edgecolor='black')
    
    # Color bars by count
    max_count = max(counts)
    colors = plt.cm.Blues([c/max_count for c in counts])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        plt.text(i, count + max_count*0.01, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to: {save_path}")
    
    plt.show()


def check_dataset_structure(data_dir):
    """
    Check and validate dataset structure
    
    Args:
        data_dir: Root directory of dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    
    stats = {
        'total_images': 0,
        'splits': {},
        'categories': set()
    }
    
    # Check for train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            split_stats = {'total': 0, 'categories': {}}
            
            # Count images per category
            for category_dir in split_path.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    stats['categories'].add(category)
                    
                    image_files = [
                        f for f in category_dir.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
                    ]
                    
                    count = len(image_files)
                    split_stats['categories'][category] = count
                    split_stats['total'] += count
            
            stats['splits'][split] = split_stats
            stats['total_images'] += split_stats['total']
    
    stats['categories'] = sorted(list(stats['categories']))
    stats['num_categories'] = len(stats['categories'])
    
    return stats


def print_dataset_info(stats):
    """
    Print dataset information
    
    Args:
        stats: Dataset statistics dictionary
    """
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"Total Images: {stats['total_images']}")
    print(f"Number of Categories: {stats['num_categories']}")
    print(f"Categories: {', '.join(stats['categories'])}")
    
    print("\nSplit Distribution:")
    for split, split_stats in stats['splits'].items():
        print(f"\n{split.upper()}:")
        print(f"  Total: {split_stats['total']} images")
        for category, count in sorted(split_stats['categories'].items()):
            print(f"    {category}: {count}")
    
    print("=" * 70)


def create_experiment_dir(base_dir='experiments'):
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
    
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f'exp_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    
    return exp_dir


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("✓ Random seed set")
    
    # Test time formatting
    print(f"✓ Time formatting: {format_time(3725)}")
    
    print("\nAll utility functions loaded successfully!")
