# GitHub CLI Commands Reference

## Project: ecommerce-AI-classifier

All GitHub CLI commands used to manage this project.

---

## 1. Repository Management

### View Repository Details
```bash
gh repo view odanree/ecommerce-AI-classifier
```

### List Repositories
```bash
gh repo list
```

---

## 2. Repository Configuration

### Update Repository Description
```bash
gh repo edit odanree/ecommerce-AI-classifier --description "Fine-tune CLIP on e-commerce product images for accurate product categorization with 92%+ accuracy"
```

### Add Topics (Tags)
```bash
gh repo edit odanree/ecommerce-AI-classifier --add-topic machine-learning,deep-learning,clip,computer-vision,transfer-learning,pytorch,e-commerce,image-classification,fine-tuning,openai
```

### Remove Topics
```bash
gh repo edit odanree/ecommerce-AI-classifier --remove-topic topic-name
```

---

## 3. Branch Protection Rules

### Set Up Branch Protection for `main`

Using GitHub API with gh cli:

```bash
gh api repos/odanree/ecommerce-AI-classifier/branches/main/protection -X PUT --input branch_protection_main.json
```

**Configuration (branch_protection_main.json):**
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": []
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

### Set Up Branch Protection for `dev`

```bash
gh api repos/odanree/ecommerce-AI-classifier/branches/dev/protection -X PUT --input branch_protection_dev.json
```

**Configuration (branch_protection_dev.json):**
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": []
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

---

## 4. Pull Requests

### List Open PRs
```bash
gh pr list --repo odanree/ecommerce-AI-classifier
```

### View PR Details
```bash
gh pr view <PR_NUMBER> --repo odanree/ecommerce-AI-classifier
```

### Create PR (from command line)
```bash
gh pr create --repo odanree/ecommerce-AI-classifier --title "Feature Title" --body "Description" --base dev --head feature/branch-name
```

### Approve PR
```bash
gh pr review <PR_NUMBER> --repo odanree/ecommerce-AI-classifier --approve
```

### Merge PR
```bash
gh pr merge <PR_NUMBER> --repo odanree/ecommerce-AI-classifier --merge
```

---

## 5. Issues

### List Issues
```bash
gh issue list --repo odanree/ecommerce-AI-classifier
```

### Create Issue
```bash
gh issue create --repo odanree/ecommerce-AI-classifier --title "Issue Title" --body "Description"
```

---

## 6. Releases

### List Releases
```bash
gh release list --repo odanree/ecommerce-AI-classifier
```

### Create Release
```bash
gh release create v1.0.0 --repo odanree/ecommerce-AI-classifier --title "Version 1.0.0" --notes "Release notes"
```

---

## 7. Useful Shortcuts

### Set Default Repository (optional)
```bash
gh repo set-default odanree/ecommerce-AI-classifier
```

After setting default, you can omit `--repo` flag:
```bash
gh pr list
gh issue list
gh release list
```

### Check Authentication Status
```bash
gh auth status
```

### View Logged-in User
```bash
gh api user -q '.login'
```

---

## 8. All Commands Used in This Project

| Command | Purpose | Status |
|---------|---------|--------|
| `gh repo edit` | Update description | ✅ Used |
| `gh repo edit --add-topic` | Add topics/tags | ✅ Used |
| `gh api /repos/.../branches/.../protection` | Set branch protection | ✅ Used |
| `gh repo view` | View repo details | ✅ Used |
| `gh repo list` | List all repos | ✅ Used |

---

## Resources

- **GitHub CLI Docs**: https://cli.github.com/manual
- **GitHub API Reference**: https://docs.github.com/en/rest
- **Branch Protection**: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches

---

Last Updated: 2025-11-13
