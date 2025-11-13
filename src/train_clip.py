"""
CLIP Fine-tuning Module for E-commerce Product Classification
Train CLIP (ViT-B/32) on labeled e-commerce product images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

from dataset_prep import create_dataloaders


class CLIPClassifier(nn.Module):
    """CLIP-based classifier for e-commerce products"""
    
    def __init__(self, num_classes: int, model_name: str = "ViT-B/32", freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of product categories
            model_name: CLIP model variant
            freeze_backbone: Whether to freeze CLIP backbone
        """
        super().__init__()
        
        self.clip_model, self.preprocess = clip.load(model_name, device="cpu")
        self.model_name = model_name
        
        # Get CLIP's output dimension
        if "ViT-B/32" in model_name or "RN50" in model_name:
            clip_dim = 512
        elif "ViT-L/14" in model_name:
            clip_dim = 768
        else:
            clip_dim = 512
        
        # Freeze CLIP backbone if specified
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images):
        """Forward pass through CLIP and classifier"""
        with torch.set_grad_enabled(self.clip_model.visual.training):
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.float()
        
        logits = self.classifier(image_features)
        return logits
    
    def encode_image(self, images):
        """Encode images to CLIP features"""
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return features


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_clip(
    data_dir: str,
    output_dir: str = "models",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    model_name: str = "ViT-B/32",
    freeze_backbone: bool = False,
    device: str = None,
    save_every: int = 5
):
    """
    Train CLIP model on e-commerce dataset
    
    Args:
        data_dir: Directory containing train/val/test splits
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        model_name: CLIP model variant
        freeze_backbone: Whether to freeze CLIP backbone
        device: Device to train on (cuda/cpu)
        save_every: Save checkpoint every N epochs
    """
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading dataset from: {data_dir}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0 if device == "cpu" else 4
    )
    
    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.get_categories())
    categories = train_loader.dataset.get_categories()
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Categories: {', '.join(categories)}")
    
    # Create model
    print(f"\nInitializing CLIP model: {model_name}")
    model = CLIPClassifier(num_classes, model_name, freeze_backbone)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': [],
        'categories': categories,
        'model_name': model_name
    }
    
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'categories': categories,
                'model_name': model_name
            }
            torch.save(checkpoint, output_path / 'clip_finetuned.pt')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'categories': categories,
                'model_name': model_name
            }
            torch.save(checkpoint, output_path / f'checkpoint_epoch_{epoch}.pt')
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to: {output_path / 'clip_finetuned.pt'}")
    print(f"Training history saved to: {output_path / 'training_history.json'}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on e-commerce dataset")
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing train/val/test splits')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--model', type=str, default='ViT-B/32',
                       choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                       help='CLIP model variant')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze CLIP backbone (only train classifier)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_clip(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        device=args.device
    )
