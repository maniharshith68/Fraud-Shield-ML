#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# fraud-shield-ml | git_init.sh
# Run ONCE after completing Phase 1 to initialise git and push
# to your GitHub repository.
#
# Usage:
#   chmod +x git_init.sh
#   ./git_init.sh
#
# Prerequisites:
#   - git installed
#   - GitHub repo created at: https://github.com/YOUR_USERNAME/fraud-shield-ml
#     (create it as EMPTY — no README, no .gitignore — let this script handle it)
# ──────────────────────────────────────────────────────────────────

set -e  # Exit immediately on error

# ── Configuration — EDIT THESE ────────────────────────────────────
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
REPO_NAME="fraud-shield-ml"
# ─────────────────────────────────────────────────────────────────

echo "=================================================="
echo " fraud-shield-ml | Git Initialisation"
echo "=================================================="

# Validate config
if [ "$GITHUB_USERNAME" = "YOUR_GITHUB_USERNAME" ]; then
    echo "ERROR: Please edit git_init.sh and set your GITHUB_USERNAME."
    exit 1
fi

# Init git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "✓ Git repository initialised"
else
    echo "✓ Git repository already exists"
fi

# Set default branch to main
git checkout -b main 2>/dev/null || git checkout main

# Stage all files
git add .

# Verify nothing sensitive is staged
if git diff --cached --name-only | grep -q "kaggle.json\|\.env"; then
    echo "ERROR: Sensitive file (kaggle.json or .env) detected in staging area."
    echo "       Check your .gitignore and run: git reset HEAD <file>"
    exit 1
fi

# First commit
git commit -m "feat: initialise fraud-shield-ml project scaffold

- Project directory structure (src/, data/, outputs/, tests/, docs/)
- Central config (config/config.yaml) with all params and paths
- Modular logging utility (loguru, rotating daily, stdout + file)
- Dot-accessible config loader (SimpleNamespace)
- IEEE-CIS dataset download script with Kaggle API integration
- Requirements.txt with pinned cross-platform dependencies
- Full Phase 1 test suite: 47 tests passing
- .gitignore excludes data, models, logs, credentials"

echo "✓ Initial commit created"

# Add remote (skip if already set)
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
if git remote get-url origin &>/dev/null; then
    echo "✓ Remote 'origin' already configured: $(git remote get-url origin)"
else
    git remote add origin "$REMOTE_URL"
    echo "✓ Remote added: $REMOTE_URL"
fi

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=================================================="
echo " ✓ Phase 1 pushed to GitHub successfully!"
echo "   https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo "=================================================="
echo ""
echo "Next step: Run the IEEE-CIS data download:"
echo "  python3 src/ingestion/download_data.py"
echo "Then: proceed to Phase 2 (Business Framing + EDA)"
