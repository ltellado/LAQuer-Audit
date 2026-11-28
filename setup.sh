#!/usr/bin/env bash
set -e

echo "=== Setting up LAQuer-Audit Environment ==="

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

# 4. Materialize and verify all benchmark datasets
echo "=== Materializing and verifying benchmark datasets (LFQA & ASQA) ==="
python3 -c "
from src.bootstrap import LAQuerAudit
LAQuerAudit.bootstrap_environment()
from src.dataset import LAQuerDataset
summary = LAQuerDataset().prepare_all()
print('Dataset preparation complete:')
print(f'  LFQA: {summary[\"lfqa_count\"]} instances at {summary[\"lfqa_path\"]}')
print(f'  ASQA: {summary[\"asqa_count\"]} instances at {summary[\"asqa_path\"]}')
"

echo "Setup complete! Please ensure OPENAI_API_KEY is exported."
