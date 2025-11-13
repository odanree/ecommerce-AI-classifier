"""
Evaluation Module for E-commerce CLIP Model
Generate metrics, confusion matrix, and sample predictions
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm
import json
import clip
from PIL import Image

from dataset_prep import create_dataloaders, EcommerceDataset, get_default_transforms
from train_clip import CLIPClassifier


def load_model(checkpoint_path: str, device: str = None):
    """
    Load trained CLIP model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model, checkpoint_info
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint['num_classes']
    model_name = checkpoint.get('model_name', 'ViT-B/32')
    
    model = CLIPClassifier(num_classes, model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Classes: {num_classes}")
    print(f"Best Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, checkpoint


def evaluate_model(model, dataloader, device, categories):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to use
        categories: List of category names
    
    Returns:
        Dictionary with predictions, labels, and paths
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_paths = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'paths': all_paths,
        'probabilities': np.array(all_probs)
    }


def compute_metrics(predictions, labels, categories):
    """
    Compute evaluation metrics
    
    Args:
        predictions: Predicted labels
        labels: True labels
        categories: List of category names
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(labels, predictions, average=None)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class': {
            categories[i]: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1_score': float(per_class_f1[i]),
                'support': int(per_class_support[i])
            }
            for i in range(len(categories))
        }
    }
    
    return metrics


def plot_confusion_matrix(cm, categories, output_path):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        categories: List of category names
        output_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=categories, yticklabels=categories,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_per_class_metrics(metrics, output_path):
    """
    Plot per-class precision, recall, and F1 scores
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save plot
    """
    categories = list(metrics['per_class'].keys())
    precisions = [metrics['per_class'][cat]['precision'] for cat in categories]
    recalls = [metrics['per_class'][cat]['recall'] for cat in categories]
    f1_scores = [metrics['per_class'][cat]['f1_score'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics to: {output_path}")


def visualize_predictions(results, categories, output_path, num_samples=16):
    """
    Visualize sample predictions
    
    Args:
        results: Evaluation results dictionary
        categories: List of category names
        output_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    predictions = results['predictions']
    labels = results['labels']
    paths = results['paths']
    probs = results['probabilities']
    
    # Get random samples
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, i in enumerate(indices):
        if idx >= len(axes):
            break
        
        try:
            img = Image.open(paths[i]).convert('RGB')
            axes[idx].imshow(img)
        except:
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        pred_label = categories[predictions[i]]
        true_label = categories[labels[i]]
        confidence = probs[i][predictions[i]] * 100
        
        color = 'green' if predictions[i] == labels[i] else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
        axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to: {output_path}")


def zero_shot_classification(images, categories, device):
    """
    Perform zero-shot classification with CLIP
    
    Args:
        images: Batch of images
        categories: List of category names
        device: Device to use
    
    Returns:
        Predictions
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Create text prompts
    text_prompts = [f"a photo of a {category}" for category in categories]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions = similarity.argmax(dim=-1)
    
    return predictions.cpu().numpy()


def compare_zero_shot_vs_finetuned(model, dataloader, categories, device, output_dir):
    """
    Compare zero-shot vs fine-tuned performance
    
    Args:
        model: Fine-tuned model
        dataloader: Test dataloader
        categories: List of categories
        device: Device to use
        output_dir: Directory to save results
    """
    print("\nComparing zero-shot vs fine-tuned performance...")
    
    # Fine-tuned predictions
    finetuned_results = evaluate_model(model, dataloader, device, categories)
    finetuned_acc = accuracy_score(finetuned_results['labels'], finetuned_results['predictions'])
    
    # Zero-shot predictions
    zeroshot_predictions = []
    zeroshot_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Zero-shot eval"):
            images = images.to(device)
            preds = zero_shot_classification(images, categories, device)
            zeroshot_predictions.extend(preds)
            zeroshot_labels.extend(labels.numpy())
    
    zeroshot_acc = accuracy_score(zeroshot_labels, zeroshot_predictions)
    
    # Create comparison
    comparison = {
        'zero_shot_accuracy': float(zeroshot_acc),
        'finetuned_accuracy': float(finetuned_acc),
        'improvement': float(finetuned_acc - zeroshot_acc),
        'improvement_percentage': float((finetuned_acc - zeroshot_acc) / zeroshot_acc * 100)
    }
    
    print(f"\nZero-shot Accuracy: {zeroshot_acc*100:.2f}%")
    print(f"Fine-tuned Accuracy: {finetuned_acc*100:.2f}%")
    print(f"Improvement: {comparison['improvement']*100:.2f}% ({comparison['improvement_percentage']:.1f}% relative)")
    
    # Save comparison
    with open(Path(output_dir) / 'zero_shot_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def run_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str = "results",
    batch_size: int = 32,
    device: str = None,
    compare_zero_shot: bool = True
):
    """
    Run complete evaluation pipeline
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing test data
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to use
        compare_zero_shot: Whether to compare with zero-shot
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    categories = checkpoint['categories']
    
    # Create dataloader
    _, _, test_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0 if device == "cpu" else 4
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device, categories)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results['predictions'], results['labels'], categories)
    
    # Print metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
    print("\nPer-class metrics:")
    for cat, cat_metrics in metrics['per_class'].items():
        print(f"  {cat}:")
        print(f"    Precision: {cat_metrics['precision']*100:.2f}%")
        print(f"    Recall: {cat_metrics['recall']*100:.2f}%")
        print(f"    F1-Score: {cat_metrics['f1_score']*100:.2f}%")
        print(f"    Support: {cat_metrics['support']}")
    
    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {output_path / 'metrics.json'}")
    
    # Confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    plot_confusion_matrix(cm, categories, output_path / 'confusion_matrix.png')
    
    # Per-class metrics plot
    plot_per_class_metrics(metrics, output_path / 'per_class_metrics.png')
    
    # Sample predictions
    visualize_predictions(results, categories, output_path / 'sample_predictions.png')
    
    # Classification report
    report = classification_report(
        results['labels'], results['predictions'],
        target_names=categories, digits=4
    )
    with open(output_path / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"Saved classification report to: {output_path / 'classification_report.txt'}")
    
    # Compare with zero-shot
    if compare_zero_shot:
        try:
            compare_zero_shot_vs_finetuned(model, test_loader, categories, device, output_dir)
        except Exception as e:
            print(f"Zero-shot comparison failed: {e}")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CLIP model")
    parser.add_argument('--model', type=str, default='models/clip_finetuned.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-zero-shot', action='store_true',
                       help='Skip zero-shot comparison')
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        compare_zero_shot=not args.no_zero_shot
    )
