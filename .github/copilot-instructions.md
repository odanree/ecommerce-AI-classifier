# Copilot Instructions: E-commerce CLIP Fine-tuning Project

## Project Overview
Fine-tune OpenAI's CLIP (ViT-B/32) on a labeled dataset of e-commerce product images (e.g., shoes, bags, shirts, electronics) for accurate, domain-specific product categorization.

## Key Features
- Dataset preparation: Download or organize product images and labels
- Fine-tuning: Train CLIP on product categories (default: ViT-B/32)
- Evaluation: Accuracy, confusion matrix, and sample predictions
- Demo: Classify a new product image and return category + confidence
- Compare zero-shot vs fine-tuned CLIP performance

## Demo Use Case
# Copilot Instructions: E-commerce CLIP Fine-tuning Project

## Project Overview
Fine-tune OpenAI's CLIP (ViT-B/32) on a labeled dataset of e-commerce product images (e.g., shoes, bags, shirts, electronics) for accurate, domain-specific product categorization.

## Key Features
- Dataset preparation: Download or organize product images and labels
- Fine-tuning: Train CLIP on product categories (default: ViT-B/32)
- Evaluation: Accuracy, confusion matrix, and sample predictions
- Demo: Classify a new product image and return category + confidence
- Compare zero-shot vs fine-tuned CLIP performance

## Demo Use Case
Upload a product image, get a precise category (e.g., "running shoes", "dress shoes", "sandals", "boots") with confidence scores. Visualize confusion matrix and sample predictions. Discuss automation for cataloging, search, and recommendations.

## Dataset
Use open e-commerce datasets (Kaggle Amazon, Shopee, etc.) or your own small set. Organize as: `dataset/<category>/<image files>`

## Workflow
1. Prepare dataset: `python src/dataset_prep.py`
2. Fine-tune CLIP: `python src/train_clip.py`
3. Evaluate: `python src/evaluate.py`
4. Demo: `python src/demo_classify.py --image path/to/image.jpg`

## Model
Default: ViT-B/32
Save fine-tuned weights as `clip_finetuned.pt` in `models/` directory

## Project Organization
**Keep root directory clean:**
- Only `README.md` and `LICENSE` should be in root
- All other documentation → `docs/`
- Source code → `src/`
- Data files → `data/`
- Models → `models/`
- Results → `results/`
- Notebooks → `notebooks/`

## Git Workflow (CRITICAL - ALWAYS FOLLOW)

### ⚠️ Branch Protection Rules
**NEVER commit directly to `main` branch!**

1. **Main Branches:**
   - `main` (production-ready, stable) - **PROTECTED: No direct commits**
   - `dev` (integration, latest development) - Merge target for features

2. **Feature Branches:**
   - Create a new branch for each feature, bugfix, or experiment
   - Naming: `feature/<name>`, `bugfix/<name>`, `experiment/<name>`
   - Examples: `feature/multi-label-support`, `bugfix/memory-leak`

3. **Development Process:**
   ```bash
   # Always start from dev
   git checkout dev
   git pull origin dev
   
   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Make changes and commit
   git add .
   git commit -m "feat: description of changes"
   
   # Push feature branch
   git push origin feature/your-feature-name
   
   # Create PR: feature/your-feature-name → dev (NOT main!)
   ```

4. **Commit Message Convention:**
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only changes
   - `style:` - Code style/formatting (no logic change)
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

5. **Pull Requests (PRs):**
   - Open PRs from feature branches to `dev` (NEVER to `main`)
   - Use descriptive PR titles
   - Request code review before merging
   - Ensure all CI/CD checks pass
   - Squash and merge or rebase for clean history

6. **Release Workflow:**
   - Merge `dev` to `main` only for stable releases
   - Tag releases (e.g., `v1.0.0`)
   - Update `CHANGELOG.md` and documentation

7. **General Tips:**
   - Keep branches focused and up to date with `dev`
   - Delete merged feature branches
   - Run tests before pushing
   - Write clear, descriptive commit messages

### When AI Creates/Modifies Code
- Always assume changes will go through PR process
- Never suggest committing directly to `main`
- Recommend creating feature branches
- Suggest appropriate commit message format
- Remind about running tests before PR

## Dataset
Use open e-commerce datasets (Kaggle Amazon, Shopee, etc.) or your own small set. Organize as: `dataset/<category>/<image files>`

## Workflow
1. Prepare dataset: `python dataset_prep.py`
2. Fine-tune CLIP: `python train_clip.py`
3. Evaluate: `python evaluate.py`
4. Demo: `python demo_classify.py --image path/to/image.jpg`

## Model
Default: ViT-B/32
Save fine-tuned weights as `clip_finetuned.pt`

## Professional Git Workflow (Recommended for All Projects)

1. **Main Branches:**
   - `main` (production-ready, stable)
   - `dev` (integration, latest development)

2. **Feature Branches:**
   - Create a new branch for each feature, bugfix, or experiment: `feature/<name>`, `bugfix/<name>`, `experiment/<name>`

3. **Commit Best Practices:**
   - Write clear, concise commit messages (imperative mood, e.g., "Add product image loader")
   - Reference issues or tasks if using a tracker
   - Commit early, commit often (atomic commits)

4. **Pull Requests (PRs):**
   - Open PRs from feature branches to `dev`
   - Use PR templates if available
   - Request code review before merging
   - Squash and merge or rebase for clean history

5. **Testing & CI:**
   - Run all tests before merging to `main`
   - Use CI/CD pipelines for linting, testing, and deployment if possible

6. **Release Workflow:**
   - Merge `dev` to `main` for releases
   - Tag releases (e.g., `v1.0.0`)
   - Update `CHANGELOG.md` and documentation

7. **General Tips:**
   - Keep branches focused and up to date with `dev`
   - Delete merged branches
   - Document workflow in `CONTRIBUTING.md` if collaborating

---

_Apply this workflow to all new projects for consistency, collaboration, and code quality._

## Portfolio Value
Demonstrates practical transfer learning, dataset curation, and robust evaluation for real-world e-commerce automation.
