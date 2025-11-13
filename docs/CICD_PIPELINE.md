# GitHub Actions CI/CD Pipeline

## Overview

This project uses GitHub Actions to automate testing, linting, security checks, and releases. The pipeline ensures code quality and reliability on every commit and pull request.

## Workflows

### 1. CI Pipeline (`ci.yml`)

**Triggers:** 
- Push to `main` or `dev` branches
- Pull requests to `main` or `dev` branches

**Jobs:**
- **Tests** (Matrix: Python 3.9, 3.10, 3.11)
  - Install dependencies
  - Run pytest with coverage
  - Upload coverage to Codecov
  
- **Linting**
  - Format check with black
  - Code style check with flake8
  - Import sorting with isort
  
- **Security**
  - Vulnerability scan with bandit
  - Dependency check with safety

**Coverage Requirements:**
- 20%+ overall coverage required to pass
- Per-module coverage tracked
- HTML reports generated

### 2. PR Validation (`pr-validation.yml`)

**Triggers:** 
- Pull requests to `main` or `dev` (opened, synchronize, reopened)

**Validation Checks:**
- ✅ PR title follows conventional commit format
- ✅ All tests pass
- ✅ Coverage maintained (20%+ minimum)
- ✅ Automatic comment with results

**Conventional Commit Format:**
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Example: feat(training): add learning rate scheduler
```

### 3. Release Workflow (`release.yml`)

**Triggers:**
- Push to `main` branch with version tags (v*.*)
- Manual workflow dispatch (for custom version)

**Release Steps:**
- Extract version from git tag or input
- Generate release notes from CHANGELOG.md
- Create GitHub Release with notes
- Tag commit automatically

**Usage:**
```bash
# Create a release
git tag v1.0.0
git push origin v1.0.0

# Or manually via GitHub Actions UI
# Actions → Release → Run workflow
```

## Local Testing Before Push

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Check Code Formatting
```bash
black src/ tests/
flake8 src/ tests/
isort src/ tests/
```

### Security Checks
```bash
bandit -r src/
safety check
```

## Monitoring

### View Workflow Status
- GitHub repository → Actions tab
- Filter by branch (main/dev)
- Click workflow run for details

### Coverage Reports
- Codecov integration tracks coverage over time
- View trends at: https://codecov.io/gh/odanree/ecommerce-AI-classifier

### Pull Request Checks
- Automated checks appear on PR page
- PR can only be merged when all checks pass (on main branch)
- Dev branch checks are informational

## Troubleshooting

### Tests Fail Locally but Pass in CI
- Check Python version matches CI matrix (3.9, 3.10, 3.11)
- Install all dependencies: `pip install -r requirements.txt`
- Clear cache: `rm -rf .pytest_cache`

### Coverage Below Threshold
- Run tests with coverage: `pytest tests/ --cov=src`
- Add tests for new code paths
- View missing coverage: `pytest --cov=src --cov-report=term-missing`

### PR Validation Fails
- Check PR title format: `<type>(<scope>): <description>`
- Ensure all new tests pass locally first
- Verify coverage is maintained above 20%

### Release Creation Issues
- Ensure version tag format is correct: `v1.0.0` (semantic versioning)
- Update CHANGELOG.md before creating release
- Release only from `main` branch (protected)

## Configuration

### Modifying Workflows

1. Edit workflow file in `.github/workflows/`
2. Commit and push to dev
3. Create PR to main for review
4. Once merged, workflow is active

### Adding New Jobs

Example:
```yaml
  new-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Step name
        run: echo "Hello from CI!"
```

### Updating Dependencies

- Workflows use `pip install -r requirements.txt`
- Update `requirements.txt` with new dependencies
- CI automatically uses updated versions

## Best Practices

✅ **Do:**
- Run tests locally before pushing
- Use descriptive PR titles (conventional commits)
- Keep workflows DRY (don't repeat configuration)
- Monitor workflow runs regularly
- Update CHANGELOG.md for releases

❌ **Don't:**
- Merge PRs with failing checks
- Force push to main branch
- Commit directly to main (use feature branches)
- Skip security checks
- Ignore coverage warnings

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Codecov Integration](https://codecov.io/)

## Contact

For questions about CI/CD setup, see [CONTRIBUTING.md](./CONTRIBUTING.md) or check the repository issues.
