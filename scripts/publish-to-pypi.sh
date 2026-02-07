#!/usr/bin/env bash
# Publish qomputing-simulator to PyPI.
# Run from repo root:  ./scripts/publish-to-pypi.sh
#
# You need a PyPI account and API token: https://pypi.org/manage/account/token/
# Option A: Run this script; when prompted, use username __token__ and password <your token>
# Option B: TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxx ./scripts/publish-to-pypi.sh

set -e
cd "$(dirname "$0")/.."

echo "Building..."
python3 -m pip install -q build twine
python3 -m build

echo ""
echo "Uploading to PyPI (you may be prompted for username/password)..."
echo "  Username: __token__"
echo "  Password: your PyPI API token (from https://pypi.org/manage/account/token/)"
echo ""
python3 -m twine upload dist/qomputing_simulator-*

echo ""
echo "Done. Anyone can now: pip install qomputing-simulator"
