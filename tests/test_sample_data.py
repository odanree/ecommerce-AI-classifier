"""
Simple end-to-end test for CLIP pipeline with sample data.
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_prep import EcommerceDataset, get_default_transforms
from train_clip import CLIPClassifier
from config import MODEL_CONFIG


def test_dataset_loading():
    """Test loading dataset from raw sample products."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        if not dataset_path.exists():
            print("✗ Sample dataset not found")
            return False
        
        # Load dataset
        dataset = EcommerceDataset(str(dataset_path), transform=get_default_transforms())
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Total images: {len(dataset)}")
        print(f"  - Categories: {', '.join(dataset.get_categories())}")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - Image shape: {sample[0].shape}")
        print(f"  - Label: {sample[1]}")
        print(f"  - Category: {dataset.get_category_name(sample[1])}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test CLIP model creation."""
    print("\n" + "="*60)
    print("TEST 2: Model Creation")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(str(dataset_path))
        num_classes = len(dataset.get_categories())
        
        # Create model
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=num_classes
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"✓ Model created successfully")
        print(f"  - Model name: {MODEL_CONFIG['model_name']}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Device: {device}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(str(dataset_path), transform=get_default_transforms())
        
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=len(dataset.get_categories())
        )
        model = model.to(device)
        model.eval()
        
        # Get sample batch (4 images)
        batch_images = []
        for i in range(min(4, len(dataset))):
            sample = dataset[i]
            batch_images.append(sample[0])
        
        batch = torch.stack(batch_images)
        
        print(f"✓ Batch created")
        print(f"  - Batch shape: {batch.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch.to(device))
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {outputs.shape}")
        print(f"  - Output logits (first sample): {outputs[0]}")
        
        # Get predictions
        predictions = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(predictions, dim=1)
        pred_confidences = torch.max(predictions, dim=1).values
        
        print(f"✓ Predictions generated")
        for i, (pred_class, confidence) in enumerate(zip(pred_classes, pred_confidences)):
            category = dataset.get_category_name(pred_class.item())
            print(f"  - Sample {i}: {category} ({confidence:.2%})")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_image_classification():
    """Test classification of a single image."""
    print("\n" + "="*60)
    print("TEST 4: Single Image Classification")
    print("="*60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dataset_path = Path("data/raw/sample_products")
        
        # Find a sample image
        sample_images = list(dataset_path.glob("**/*.png"))
        if not sample_images:
            print("✗ No sample images found")
            return False
        
        sample_image_path = sample_images[0]
        print(f"Using sample image: {sample_image_path.name}")
        
        # Load image
        image = Image.open(sample_image_path).convert('RGB')
        transform = get_default_transforms()
        image_tensor = transform(image)
        
        print(f"✓ Image loaded and transformed")
        print(f"  - Image shape: {image_tensor.shape}")
        
        # Create model and classify
        dataset = EcommerceDataset(str(dataset_path))
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=len(dataset.get_categories())
        )
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(device))
            predictions = torch.softmax(output, dim=1)[0]
            top_k = torch.topk(predictions, 3)
        
        print(f"✓ Classification successful")
        print(f"  - Top 3 predictions:")
        for prob, idx in zip(top_k.values, top_k.indices):
            category = dataset.get_category_name(idx.item())
            print(f"    - {category}: {prob:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("E-COMMERCE CLIP - SAMPLE DATA TEST")
    print("="*60)
    
    results = {
        "Dataset Loading": test_dataset_loading(),
        "Model Creation": test_model_creation(),
        "Forward Pass": test_forward_pass(),
        "Single Image Classification": test_single_image_classification(),
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Pipeline is working with sample data.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
    
    print("="*60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
