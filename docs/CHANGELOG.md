# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD pipeline for automated testing and linting
- PR validation workflow with conventional commit format checking
- Release workflow for automated GitHub releases
- Comprehensive unit test suite (66 tests) with coverage reporting

### Changed
- Updated testing infrastructure with pytest and coverage tools

## [1.0.0] - 2025-11-13

### Added
- Initial release of e-commerce CLIP fine-tuning project
- Core implementation modules:
  - `dataset_prep.py`: Dataset loading, splitting, and augmentation
  - `train_clip.py`: CLIP model fine-tuning with training loop
  - `evaluate.py`: Evaluation metrics, confusion matrix, and visualization
  - `demo_classify.py`: Single image classification demo
  - `config.py`: Centralized configuration management
  - `utils.py`: Utility functions for image processing and helpers
- Comprehensive test suite (66 tests) covering all core modules
- Sample data generation script for quick testing
- Full documentation (README, CONTRIBUTING, technical guides)
- Git workflow with branch protection on main branch
- Support for multiple Python versions (3.9, 3.10, 3.11)

### Features
- Fine-tune OpenAI CLIP (ViT-B/32) on custom e-commerce product categories
- Evaluate model performance with accuracy, precision, recall, F1 score
- Generate confusion matrix and visualizations
- Classify individual product images with confidence scores
- Automatic model checkpointing during training

### Testing
- 18 unit tests for dataset preparation module
- 19 unit tests for CLIP training module
- 19 unit tests for evaluation module
- 4 end-to-end pipeline tests
- 6 integration tests
- 23% code coverage on core modules

### Documentation
- README with project overview and quick start guide
- CONTRIBUTING guide with Git workflow instructions
- Copilot instructions for AI-assisted development
- GitHub Actions CI/CD pipeline documentation

### CI/CD
- Automated testing on Python 3.9, 3.10, 3.11
- Code coverage reporting with Codecov integration
- Linting with flake8, black, and isort
- Security scanning with bandit and safety
- PR validation with conventional commit format checking
- Automated release creation on version tags

---

## Contributing

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
