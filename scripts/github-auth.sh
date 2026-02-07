#!/usr/bin/env bash
# GitHub authentication: open browser, get one-time code, then Git push works.
# Run from repo root:  ./scripts/github-auth.sh

set -e

echo "=== GitHub authentication (browser redirect + passcode) ==="
echo ""

if command -v gh &>/dev/null; then
  echo "Using GitHub CLI (gh). You will be redirected to GitHub in your browser."
  echo "Log in there and paste the one-time code back here when prompted."
  echo ""
  gh auth login --web --git-protocol https
  echo ""
  echo "Done. You can now run:  git push origin main"
else
  echo "GitHub CLI (gh) is not installed. Two options:"
  echo ""
  echo "Option A - Install GitHub CLI (recommended; uses browser redirect):"
  echo "  macOS (Homebrew):  brew install gh"
  echo "  Then run:  gh auth login --web --git-protocol https"
  echo "  Then run:  ./scripts/github-auth.sh  again, or  git push origin main"
  echo ""
  echo "Option B - Use a Personal Access Token (no browser):"
  echo "  1. Open: https://github.com/settings/tokens"
  echo "  2. Generate new token (classic), enable 'repo' scope"
  echo "  3. When Git asks for a password, paste the token (not your GitHub password)"
  echo ""
  exit 1
fi
