#$ -S /bin/bash
#$ -l tmem=16G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=4:00:00
# Recompute phase-1 metrics of the final runs of one dataset from saved heads (CPU): qsub -N eval_<dataset> jobs/evaluate.qsub.sh <dataset>
set -eo pipefail
DATASET=$1
[ -n "$DATASET" ] || { echo "usage: qsub jobs/evaluate.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
python3 -m b2j.evaluate "$@"
date
