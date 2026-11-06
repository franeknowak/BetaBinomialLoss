#$ -S /bin/bash
#$ -l tmem=8G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=1:00:00
#$ -N dataset_examples
# Sample pack for notebooks/datasets_overview.ipynb: label tables + example thumbnails of all 5 datasets.
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
export MPLBACKEND=Agg
python3 -m b2j.dataset_examples --out results/dataset_examples
du -sh results/dataset_examples; find results/dataset_examples -type f | wc -l
date
