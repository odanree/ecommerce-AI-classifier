"""
Dataset Preparation Module for E-commerce CLIP Fine-tuning
Handles downloading, organizing, and loading product images by category
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json


class EcommerceDataset(Dataset):
    """Custom dataset for e-commerce product images"""
    
    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        """
        Args:
            root_dir: Root directory containing category folders
            transform: Optional transform to be applied on images
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        self.images = []
        self.labels = []
        self.category_to_idx = {}
        self.idx_to_category = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images and create category mappings"""
        categories = sorted([d for d in os.listdir(self.root_dir) 
                           if os.path.isdir(os.path.join(self.root_dir, d))])
        
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        for category in categories:
            category_path = self.root_dir / category
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            for img_file in image_files:
                self.images.append(str(category_path / img_file))
                self.labels.append(self.category_to_idx[category])
        
        print(f"Loaded {len(self.images)} images from {len(categories)} categories")
        print(f"Categories: {', '.join(categories)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label, img_path
    
    def get_category_name(self, idx: int) -> str:
        """Get category name from label index"""
        return self.idx_to_category.get(idx, "Unknown")
    
    def get_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(self.category_to_idx.keys())


def get_default_transforms(img_size: int = 224):
    """Get default image transforms for CLIP"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Directory containing category folders with images
        output_dir: Directory to save split datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    categories = [d for d in os.listdir(source_path) 
                 if os.path.isdir(source_path / d)]
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for category in categories:
        category_path = source_path / category
        images = [f for f in os.listdir(category_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        random.shuffle(images)
        
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split, split_images in splits.items():
            split_category_path = output_path / split / category
            split_category_path.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                src = category_path / img
                dst = split_category_path / img
                shutil.copy2(src, dst)
            
            stats[split] += len(split_images)
    
    print(f"\nDataset split complete:")
    print(f"  Train: {stats['train']} images")
    print(f"  Val: {stats['val']} images")
    print(f"  Test: {stats['test']} images")
    print(f"  Total: {sum(stats.values())} images")
    
    # Save split info
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump({
            'stats': stats,
            'categories': categories,
            'ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }, f, indent=2)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing train/val/test splits
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        img_size: Image size for transforms
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transform = get_default_transforms(img_size)
    
    train_dataset = EcommerceDataset(
        os.path.join(data_dir, 'train'),
        transform=transform,
        split='train'
    )
    
    val_dataset = EcommerceDataset(
        os.path.join(data_dir, 'val'),
        transform=transform,
        split='val'
    )
    
    test_dataset = EcommerceDataset(
        os.path.join(data_dir, 'test'),
        transform=transform,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare e-commerce dataset")
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory with category folders')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    split_dataset(
        args.source,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
