# Contributing to E-commerce CLIP Classifier

Thank you for your interest in contributing! This document provides guidelines and best practices for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## 🚀 Getting Started

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Fork via GitHub UI, then clone your fork
   git clone https://github.com/YOUR_USERNAME/ecommerce-AI-classifier.git
   cd ecommerce-AI-classifier
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/ecommerce-AI-classifier.git
   ```

3. **Create development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Verify setup**
   ```bash
   pytest tests/
   ```

## 🔄 Development Workflow

### Branch Structure

- **`main`**: Production-ready, stable code
- **`dev`**: Integration branch for latest development
- **`feature/<name>`**: New features
- **`bugfix/<name>`**: Bug fixes
- **`experiment/<name>`**: Experimental changes

### Creating a Feature Branch

1. **Update your local repository**
   ```bash
   git checkout dev
   git pull upstream dev
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

### Commit Message Guidelines

Follow the imperative mood and be descriptive:

**Good examples:**
- ✅ `Add support for ViT-L/14 model variant`
- ✅ `Fix data loader memory leak in training loop`
- ✅ `Update evaluation metrics visualization`
- ✅ `Refactor config module for better modularity`

**Bad examples:**
- ❌ `Updated stuff`
- ❌ `Fixed bug`
- ❌ `Changes`

**Format:**
```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 📝 Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped and sorted (use `isort`)
- **Docstrings**: Google-style docstrings

### Code Formatting

Use automated formatters:

```bash
# Format code with black
black src/

# Sort imports
isort src/

# Check style
flake8 src/
```

### Type Hints

Use type hints for function signatures:

```python
def classify_image(
    image_path: str,
    model: nn.Module,
    categories: List[str],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Classify a product image.
    
    Args:
        image_path: Path to image file
        model: Trained model
        categories: List of category names
        device: Device to use for inference
    
    Returns:
        Dictionary containing predictions and confidence scores
    """
    pass
```

### Docstring Standards

Use Google-style docstrings:

```python
def train_model(data_loader, epochs, learning_rate):
    """
    Train the CLIP model on e-commerce dataset.
    
    Args:
        data_loader (DataLoader): Training data loader
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        tuple: Trained model and training history
    
    Raises:
        ValueError: If epochs <= 0 or learning_rate <= 0
    
    Examples:
        >>> model, history = train_model(loader, epochs=10, learning_rate=1e-4)
    """
    pass
```

## 🔍 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
   ```bash
   pytest tests/
   ```

2. **Format your code**
   ```bash
   black src/
   isort src/
   ```

3. **Update documentation**
   - Update README.md if adding features
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md (if exists)

4. **Commit and push**
   ```bash
   git push origin feature/your-feature-name
   ```

### Submitting a Pull Request

1. **Create PR to dev branch**
   ```bash
   gh pr create --base dev --head feature/your-feature-name
   ```

2. **Fill out PR template**
   - Description of changes
   - Related issues (if any)
   - Testing done
   - Screenshots (for UI changes)

3. **PR Title Format**
   ```
   [Type] Brief description of changes
   ```
   
   Examples:
   - `[Feature] Add multi-label classification support`
   - `[Fix] Resolve memory leak in data loader`
   - `[Docs] Update installation instructions`

### PR Review and Merge Process

1. **Automated checks**: CI/CD must pass
2. **Code review**: At least one maintainer approval required
3. **Address feedback**: Make requested changes
4. **Squash and merge to dev**
   ```bash
   gh pr merge <PR_NUMBER> --squash
   ```

### Release Workflow (dev → main)

1. **Create PR from dev to main** (for releases only)
   ```bash
   git checkout main
   git pull origin main
   gh pr create --base main --head dev --title "release: v1.0.0"
   ```

2. **Get approval and squash merge to main**
   ```bash
   gh pr merge <PR_NUMBER> --squash
   ```

3. **Tag the release**
   ```bash
   gh release create v1.0.0 --target main
   ```

4. **Synchronize main back to dev**
   ```bash
   git checkout dev
   git pull origin main
   git push origin dev
   ```

### Complete Git Workflow

```
dev → feature/branch → PR → dev (squash merge)
   → (for releases) dev → PR → main (squash merge)
   → sync main → dev
```

### Submitting a Pull Request

1. **Go to GitHub and create PR**
   - Base branch: `dev` (not `main`)
   - Compare branch: your feature branch

2. **Fill out PR template**
   - Description of changes
   - Related issues (if any)
   - Testing done
   - Screenshots (for UI changes)

3. **PR Title Format**
   ```
   [Type] Brief description of changes
   ```
   
   Examples:
   - `[Feature] Add multi-label classification support`
   - `[Fix] Resolve memory leak in data loader`
   - `[Docs] Update installation instructions`

### PR Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Related Issues
Fixes #(issue number)

## Testing
- [ ] All tests pass
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks**: CI/CD must pass
2. **Code review**: At least one maintainer approval required
3. **Address feedback**: Make requested changes
4. **Merge**: Maintainer will merge after approval

### Merge Strategy

- **Squash and merge**: For feature branches (keeps history clean)
- **Rebase and merge**: For small fixes
- **Merge commit**: For significant features (preserves history)

## 🧪 Testing Guidelines

### Writing Tests

Place tests in `tests/` directory:

```
tests/
├── test_dataset_prep.py
├── test_train_clip.py
├── test_evaluate.py
└── test_utils.py
```

### Test Structure

```python
import pytest
from src.utils import set_seed, count_parameters

class TestUtils:
    """Test utility functions"""
    
    def test_set_seed(self):
        """Test random seed setting"""
        set_seed(42)
        # Assert reproducibility
        
    def test_count_parameters(self):
        """Test parameter counting"""
        # Create mock model
        # Assert correct counts
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include type hints
- Provide examples in docstrings when helpful

### README Updates

When adding features, update:
- Installation instructions (if dependencies added)
- Usage examples
- Feature list
- Configuration options

### Creating Examples

Add examples in `examples/` or `notebooks/`:
- Jupyter notebooks for tutorials
- Python scripts for common use cases
- Include comments explaining each step

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Verify it's reproducible

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command X
2. With parameters Y
3. See error

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8]

**Additional context**
Any other relevant information
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Implementation**
How might this be implemented?

**Alternatives Considered**
What alternatives have you considered?
```

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Chat**: Join our Discord/Slack (if available)
- **Email**: contact@example.com

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for contributing to E-commerce CLIP Classifier!** 🎉
