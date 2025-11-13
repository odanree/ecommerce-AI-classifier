"""
Demo Classification Script for E-commerce CLIP Model
Classify new product images and return category with confidence scores
"""

import os
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json

from train_clip import CLIPClassifier
from dataset_prep import get_default_transforms


def load_model(checkpoint_path: str, device: str = None):
    """
    Load trained CLIP model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model, categories, model_info
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint['num_classes']
    categories = checkpoint['categories']
    model_name = checkpoint.get('model_name', 'ViT-B/32')
    
    model = CLIPClassifier(num_classes, model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    model_info = {
        'model_name': model_name,
        'num_classes': num_classes,
        'val_acc': checkpoint.get('val_acc', None),
        'epoch': checkpoint.get('epoch', None)
    }
    
    return model, categories, model_info


def classify_image(
    image_path: str,
    model,
    categories: list,
    device: str,
    top_k: int = 5
):
    """
    Classify a single product image
    
    Args:
        image_path: Path to image file
        model: Trained model
        categories: List of category names
        device: Device to use
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and probabilities
    """
    # Load and preprocess image
    transform = get_default_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(categories)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'category': categories[idx.item()],
            'confidence': float(prob.item()),
            'confidence_percent': float(prob.item() * 100)
        })
    
    return {
        'image_path': image_path,
        'top_prediction': predictions[0],
        'all_predictions': predictions
    }


def visualize_prediction(
    image_path: str,
    result: dict,
    output_path: str = None,
    show: bool = True
):
    """
    Visualize classification result
    
    Args:
        image_path: Path to image
        result: Classification result dictionary
        output_path: Path to save visualization
        show: Whether to display the plot
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    top_pred = result['top_prediction']
    ax1.set_title(
        f"Predicted: {top_pred['category']}\nConfidence: {top_pred['confidence_percent']:.2f}%",
        fontsize=14,
        fontweight='bold',
        color='green'
    )
    
    # Display top predictions as horizontal bar chart
    categories = [pred['category'] for pred in result['all_predictions']]
    confidences = [pred['confidence_percent'] for pred in result['all_predictions']]
    
    y_pos = range(len(categories))
    colors = plt.cm.Blues([(c/100)**0.5 for c in confidences])
    
    ax2.barh(y_pos, confidences, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories)
    ax2.invert_yaxis()
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add confidence values on bars
    for i, (cat, conf) in enumerate(zip(categories, confidences)):
        ax2.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def classify_batch(
    image_paths: list,
    model,
    categories: list,
    device: str,
    output_dir: str = None,
    top_k: int = 5
):
    """
    Classify multiple images
    
    Args:
        image_paths: List of image paths
        model: Trained model
        categories: List of category names
        device: Device to use
        output_dir: Directory to save results
        top_k: Number of top predictions
    
    Returns:
        List of classification results
    """
    results = []
    
    for image_path in image_paths:
        print(f"\nClassifying: {image_path}")
        try:
            result = classify_image(image_path, model, categories, device, top_k)
            results.append(result)
            
            # Print results
            print(f"  Top prediction: {result['top_prediction']['category']} "
                  f"({result['top_prediction']['confidence_percent']:.2f}%)")
            
            if output_dir:
                # Save visualization
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                img_name = Path(image_path).stem
                vis_path = output_path / f"{img_name}_prediction.png"
                visualize_prediction(image_path, result, str(vis_path), show=False)
        
        except Exception as e:
            print(f"  Error classifying {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    # Save batch results
    if output_dir:
        output_path = Path(output_dir)
        with open(output_path / 'batch_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved batch results to: {output_path / 'batch_results.json'}")
    
    return results


def demo_interactive(model_path: str, device: str = None):
    """
    Interactive demo mode
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("E-COMMERCE PRODUCT CLASSIFIER - INTERACTIVE DEMO")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, categories, model_info = load_model(model_path, device)
    
    print(f"\nModel: {model_info['model_name']}")
    print(f"Categories: {', '.join(categories)}")
    if model_info['val_acc']:
        print(f"Validation Accuracy: {model_info['val_acc']:.2f}%")
    
    print("\n" + "=" * 70)
    print("Enter image path to classify (or 'quit' to exit)")
    print("=" * 70)
    
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting demo...")
            break
        
        if not image_path or not os.path.exists(image_path):
            print("Invalid path. Please enter a valid image path.")
            continue
        
        try:
            # Classify
            result = classify_image(image_path, model, categories, device, top_k=5)
            
            # Print results
            print("\n" + "-" * 70)
            print("CLASSIFICATION RESULTS")
            print("-" * 70)
            print(f"Top Prediction: {result['top_prediction']['category']}")
            print(f"Confidence: {result['top_prediction']['confidence_percent']:.2f}%")
            print("\nAll Top-5 Predictions:")
            for i, pred in enumerate(result['all_predictions'], 1):
                print(f"  {i}. {pred['category']}: {pred['confidence_percent']:.2f}%")
            
            # Visualize
            visualize_prediction(image_path, result, output_path=None, show=True)
        
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify e-commerce product images with fine-tuned CLIP"
    )
    parser.add_argument('--model', type=str, default='models/clip_finetuned.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image to classify')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                       help='Paths to multiple images to classify')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory containing images to classify')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                       help='Directory to save results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    if args.interactive:
        demo_interactive(args.model, args.device)
    
    elif args.image:
        # Single image classification
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading model...")
        model, categories, model_info = load_model(args.model, args.device)
        
        print(f"\nClassifying: {args.image}")
        result = classify_image(args.image, model, categories, args.device, args.top_k)
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION RESULTS")
        print("=" * 70)
        print(f"Image: {args.image}")
        print(f"Top Prediction: {result['top_prediction']['category']}")
        print(f"Confidence: {result['top_prediction']['confidence_percent']:.2f}%")
        print(f"\nTop-{args.top_k} Predictions:")
        for i, pred in enumerate(result['all_predictions'], 1):
            print(f"  {i}. {pred['category']}: {pred['confidence_percent']:.2f}%")
        
        # Save/show visualization
        output_path = None
        if args.save_viz:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(args.image).stem}_prediction.png"
        
        visualize_prediction(args.image, result, output_path, show=True)
    
    elif args.images or args.image_dir:
        # Batch classification
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading model...")
        model, categories, model_info = load_model(args.model, args.device)
        
        # Collect image paths
        image_paths = []
        if args.images:
            image_paths.extend(args.images)
        if args.image_dir:
            img_dir = Path(args.image_dir)
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                image_paths.extend(str(p) for p in img_dir.glob(f'*{ext}'))
                image_paths.extend(str(p) for p in img_dir.glob(f'*{ext.upper()}'))
        
        print(f"\nClassifying {len(image_paths)} images...")
        
        output_dir = args.output_dir if args.save_viz else None
        results = classify_batch(image_paths, model, categories, args.device, output_dir, args.top_k)
        
        print("\n" + "=" * 70)
        print(f"Classified {len(results)} images")
        if output_dir:
            print(f"Results saved to: {output_dir}")
        print("=" * 70)
    
    else:
        print("Please specify --image, --images, --image-dir, or --interactive")
        print("Run with --help for usage information")
