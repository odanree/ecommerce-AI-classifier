"""
End-to-end testing script for CLIP fine-tuning pipeline.
Tests dataset preparation, training, evaluation, and demo classification.
"""

import os
import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG
from dataset_prep import EcommerceDataset, create_dataloaders
from train_clip import CLIPClassifier, train_epoch, validate
from evaluate import evaluate_model
from demo_classify import classify_image


def test_dataset_preparation():
    """Test dataset preparation and data loading."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Preparation")
    print("="*60)
    
    dataset_path = Path("data/raw/sample_products")
    if not dataset_path.exists():
        print("✗ Sample dataset not found. Run: python src/create_sample_data.py")
        return False
    
    try:
        # Create dataset
        dataset = EcommerceDataset(dataset_path)
        print(f"✓ Dataset created with {len(dataset)} images")
        print(f"  Categories: {dataset.categories}")
        
        # Check data
        if len(dataset) == 0:
            print("✗ Dataset is empty!")
            return False
        
        # Get sample
        sample = dataset[0]
        print(f"✓ Sample loaded:")
        print(f"  - Image shape: {sample['image'].shape if isinstance(sample['image'], torch.Tensor) else 'N/A'}")
        print(f"  - Label: {sample['label']}")
        print(f"  - Category: {sample['category']}")
        
        return True
    except Exception as e:
        print(f"✗ Error in dataset preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """Test data loader creation and batching."""
    print("\n" + "="*60)
    print("TEST 2: Data Loaders")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(dataset_path)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, 
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        print(f"✓ Data loaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Check batch
        for batch in train_loader:
            print(f"✓ Sample batch loaded:")
            print(f"  - Batch size: {len(batch['images'])}")
            print(f"  - Image shape: {batch['images'].shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Error in data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test CLIP model initialization."""
    print("\n" + "="*60)
    print("TEST 3: Model Initialization")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(dataset_path)
        num_classes = len(dataset.categories)
        
        # Initialize model
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=num_classes
        )
        
        print(f"✓ Model initialized")
        print(f"  - Model: {MODEL_CONFIG['model_name']}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Device: {model.device}")
        
        # Check if model has required attributes
        assert hasattr(model, 'model'), "Model should have 'model' attribute"
        assert hasattr(model, 'classifier'), "Model should have 'classifier' attribute"
        
        return True
    except Exception as e:
        print(f"✗ Error in model initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("TEST 4: Training Step")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(dataset_path)
        train_loader, _, _ = create_dataloaders(dataset, batch_size=4)
        
        # Initialize model
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=len(dataset.categories)
        )
        
        # Single training epoch
        print("Running one training epoch...")
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
            device=model.device,
            epoch=1
        )
        
        print(f"✓ Training step completed")
        print(f"  - Training loss: {train_loss:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_step():
    """Test a validation step."""
    print("\n" + "="*60)
    print("TEST 5: Validation Step")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        dataset = EcommerceDataset(dataset_path)
        _, val_loader, _ = create_dataloaders(dataset, batch_size=4)
        
        # Initialize model
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=len(dataset.categories)
        )
        model.eval()
        
        # Validation
        print("Running validation...")
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            device=model.device
        )
        
        print(f"✓ Validation step completed")
        print(f"  - Validation loss: {val_loss:.4f}")
        print(f"  - Validation accuracy: {val_acc:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ Error in validation step: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demo_classification():
    """Test demo classification on sample image."""
    print("\n" + "="*60)
    print("TEST 6: Demo Classification")
    print("="*60)
    
    try:
        dataset_path = Path("data/raw/sample_products")
        
        # Find a sample image
        sample_images = list(dataset_path.glob("**/*.png"))
        if not sample_images:
            print("✗ No sample images found")
            return False
        
        sample_image = sample_images[0]
        print(f"Using sample image: {sample_image}")
        
        # Create a simple model for demo (zero-shot since we didn't train)
        dataset = EcommerceDataset(dataset_path)
        model = CLIPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_classes=len(dataset.categories)
        )
        
        # Classify image (will use zero-shot CLIP)
        print("Classifying image...")
        result = classify_image(
            model=model,
            image_path=str(sample_image),
            categories=dataset.categories,
            device=model.device
        )
        
        print(f"✓ Classification completed")
        print(f"  - Predicted class: {result['predicted_class']}")
        print(f"  - Confidence: {result['confidence']:.2%}")
        print(f"  - Top 3 predictions:")
        for i, (cat, score) in enumerate(result['scores'][:3], 1):
            print(f"    {i}. {cat}: {score:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ Error in demo classification: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_summary(results):
    """Print final test summary."""
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
        print("\n✓ All tests passed! Pipeline is working correctly.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the errors above.")
    
    print("="*60)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("E-COMMERCE CLIP PIPELINE - END-TO-END TEST")
    print("="*60)
    
    results = {
        "Dataset Preparation": test_dataset_preparation(),
        "Data Loaders": test_data_loaders(),
        "Model Initialization": test_model_initialization(),
        "Training Step": test_training_step(),
        "Validation Step": test_validation_step(),
        "Demo Classification": test_demo_classification(),
    }
    
    print_final_summary(results)
    
    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
