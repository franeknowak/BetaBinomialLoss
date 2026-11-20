#$ -S /bin/bash
#$ -l tmem=8G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=1:00:00
#$ -N install_transformers
# Adds transformers (requirements.txt version) to dino_env without changing any installed package: the current
# versions are passed as constraints, so pip fails instead of up- or downgrading them.
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
VERSION=$(grep -E '^transformers==' requirements.txt | cut -d' ' -f1)
CONSTRAINTS=$(mktemp)
pip freeze --exclude-editable > "$CONSTRAINTS"
pip install "$VERSION" -c "$CONSTRAINTS"
pip freeze --exclude-editable | diff "$CONSTRAINTS" - || true
python3 -c "import torch, transformers, huggingface_hub, timm; from transformers import DINOv3ViTModel; print('ok', torch.__version__, transformers.__version__, huggingface_hub.__version__, timm.__version__)"
rm -f "$CONSTRAINTS"
date
