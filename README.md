# E-commerce Product Classification with Fine-tuned CLIP

Fine-tune OpenAI's CLIP (ViT-B/32) model on labeled e-commerce product images for accurate, domain-specific product categorization. This project demonstrates practical transfer learning, dataset curation, and robust evaluation for real-world e-commerce automation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

This project fine-tunes CLIP on e-commerce product images to achieve superior classification accuracy compared to zero-shot learning. The fine-tuned model can accurately categorize products into specific classes (e.g., "running shoes", "dress shoes", "sandals", "boots") with high confidence.

### Key Features

- **Dataset Preparation**: Automated download and organization of product images by category
- **Fine-tuning**: Train CLIP on product categories with customizable hyperparameters
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and sample predictions
- **Interactive Demo**: Classify new product images and visualize results with confidence scores
- **Zero-shot Comparison**: Benchmark fine-tuned model against zero-shot CLIP

### Use Cases

- **Automated Product Cataloging**: Instantly categorize thousands of products
- **Enhanced Search**: Improve product discovery with accurate categorization
- **Recommendation Systems**: Power better recommendations through precise classification
- **Quality Control**: Verify product listings match their categories

## 📁 Project Structure

```
ecommerce-AI-classifier/
├── src/                      # Source code
│   ├── dataset_prep.py      # Dataset preparation and loading
│   ├── train_clip.py         # CLIP fine-tuning script
│   ├── evaluate.py           # Evaluation and metrics
│   ├── demo_classify.py      # Demo classification script
│   ├── config.py             # Configuration settings
│   └── utils.py              # Utility functions
├── data/                     # Data storage
│   ├── raw/                  # Raw dataset
│   └── processed/            # Train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── models/                   # Saved model checkpoints
├── results/                  # Evaluation results
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── docs/                     # Documentation
│   └── CONTRIBUTING.md       # Contribution guidelines
├── .github/
│   └── copilot-instructions.md
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                 # This file (main documentation)
```

**Note**: Keep only `README.md` and `LICENSE` in the root directory. All other documentation goes in `docs/`, and project files in their respective subdirectories.

## 🔀 Git Workflow

**⚠️ IMPORTANT: DO NOT commit directly to `main` branch!**

This project follows a professional Git workflow:

### Branch Structure
- **`main`**: Production-ready, stable code (for releases only)
- **`dev`**: Integration branch for development (merge target for features)
- **`feature/<name>`**: New features (e.g., `feature/multi-label-support`)
- **`bugfix/<name>`**: Bug fixes (e.g., `bugfix/memory-leak`)
- **`experiment/<name>`**: Experimental changes

### Development Process

1. **Start from dev branch**
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   **Commit message format**:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

4. **Push to GitHub and create PR to dev**
   ```bash
   git push origin feature/your-feature-name
   gh pr create --base dev --head feature/your-feature-name
   ```

5. **Code review and squash merge to dev**
   ```bash
   gh pr merge <PR_NUMBER> --squash
   ```

6. **For releases: Create PR from dev → main**
   ```bash
   git checkout main
   git pull origin main
   gh pr create --base main --head dev --title "release: v1.0.0"
   
   # After approval, squash merge to main
   gh pr merge <PR_NUMBER> --squash
   
   # Tag the release
   gh release create v1.0.0 --target main
   ```

7. **Synchronize main → dev**
   ```bash
   git checkout dev
   git pull origin main
   git push origin dev
   ```

### Complete Workflow
```
dev → feature/branch → PR → dev (squash merge)
   → PR → main (squash merge on release)
   → sync main → dev
```

For detailed contribution guidelines, see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## �🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-AI-classifier.git
   cd ecommerce-AI-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

1. **Organize your dataset**
   
   Place your e-commerce product images in the following structure:
   ```
   data/raw/
   ├── running_shoes/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── dress_shoes/
   │   ├── image1.jpg
   │   └── ...
   ├── sandals/
   └── boots/
   ```

   **Dataset Sources** (optional):
   - [Kaggle Amazon Products](https://www.kaggle.com/datasets)
   - [Shopee Product Dataset](https://www.kaggle.com/datasets)
   - Your own curated product images

2. **Split dataset into train/val/test**
   ```bash
   python src/dataset_prep.py --source data/raw --output data/processed
   ```

### Training

Train the CLIP model on your dataset:

```bash
python src/train_clip.py --data-dir data/processed --epochs 10 --batch-size 32
```

**Training options:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--model`: CLIP variant (default: ViT-B/32)
- `--freeze-backbone`: Freeze CLIP backbone (only train classifier)
- `--device`: Device (cuda/cpu)

### Evaluation

Evaluate the trained model:

```bash
python src/evaluate.py --model models/clip_finetuned.pt --data-dir data/processed
```

This generates:
- Accuracy, precision, recall, F1 scores
- Confusion matrix visualization
- Per-class metrics
- Sample prediction visualizations
- Zero-shot vs fine-tuned comparison

### Demo: Classify New Images

**Single image:**
```bash
python src/demo_classify.py --image path/to/product.jpg --save-viz
```

**Multiple images:**
```bash
python src/demo_classify.py --images img1.jpg img2.jpg img3.jpg --save-viz
```

**Directory of images:**
```bash
python src/demo_classify.py --image-dir path/to/images/ --save-viz
```

**Interactive mode:**
```bash
python src/demo_classify.py --interactive
```

## 📊 Results

### Performance Comparison

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Zero-shot CLIP | ~65% | ~63% | ~65% | ~64% |
| Fine-tuned CLIP | **~92%** | **~91%** | **~92%** | **~91%** |

*Results may vary based on dataset and training configuration*

### Sample Output

```
CLASSIFICATION RESULTS
======================================================================
Image: running_shoe_001.jpg
Top Prediction: running shoes
Confidence: 96.3%

Top-5 Predictions:
  1. running shoes: 96.3%
  2. athletic shoes: 2.1%
  3. sneakers: 1.2%
  4. training shoes: 0.3%
  5. walking shoes: 0.1%
```

## 🛠️ Configuration

Customize training and evaluation in `src/config.py`:

```python
MODEL_CONFIG = {
    'model_name': 'ViT-B/32',
    'freeze_backbone': False,
}

TRAINING_CONFIG = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-4,
}
```

## 📈 Monitoring Training

Training outputs:
- Real-time progress bars with loss and accuracy
- Checkpoints saved every 5 epochs
- Best model saved based on validation accuracy
- Training history saved as JSON

Example output:
```
Epoch 5/10
----------------------------------------------------------------------
Training: 100%|████████| 150/150 [02:34<00:00, loss=0.1234, acc=95.2%]
Validation: 100%|██████| 50/50 [00:45<00:00, loss=0.1567, acc=93.8%]

Epoch 5 Summary:
  Train Loss: 0.1234 | Train Acc: 95.2%
  Val Loss: 0.1567 | Val Acc: 93.8%
  ✓ Saved best model (Val Acc: 93.8%)
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{ecommerce-clip-classifier,
  author = {Your Name},
  title = {E-commerce Product Classification with Fine-tuned CLIP},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ecommerce-AI-classifier}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

**Key Points:**
- Fork the repository and create feature branches
- Never commit directly to `main` branch
- All changes must go through Pull Requests to `dev`
- Follow the commit message conventions
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the pre-trained model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- E-commerce dataset providers

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🗺️ Roadmap

- [ ] Add support for more CLIP variants (ViT-L/14, RN101)
- [ ] Implement data augmentation strategies
- [ ] Add multi-label classification support
- [ ] Create web interface for demo
- [ ] Add model deployment guide (Docker, FastAPI)
- [ ] Integrate with MLflow for experiment tracking
- [ ] Add few-shot learning capabilities

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ for e-commerce automation**
