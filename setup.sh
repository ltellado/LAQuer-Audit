#!/usr/bin/env bash
set -e

echo "=== Setting up LAQuer-Post-Hoc-Audit Environment ==="

# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone and pin official LAQuer repository for base prompts and LFQA splits
if [ ! -d "LAQuer" ]; then
    echo "Cloning official LAQuer repository..."
    git clone https://github.com/eranhirs/LAQuer.git
    cd LAQuer
    git checkout a4affb2f39e781463a7d16cddc8baea2d162d314
    cd ..
    echo "LAQuer pinned to commit a4affb2 successfully."
else
    echo "LAQuer repository already present."
fi

# 3. Create required runtime directories
mkdir -p results data

echo "Setup complete! Please ensure OPENAI_API_KEY is exported."
