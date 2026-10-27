#$ -S /bin/bash
#$ -l tmem=32G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=24:00:00
# Tuning grid of one dataset (36 runs, seed 0), then selection: qsub -N tune_<dataset> [-hold_jid embed_<dataset>] jobs/tune.qsub.sh <dataset>
# Grid extension: qsub -v EXTRA_LRS=5e-3 ... runs only the extra learning rate(s), then selects again.
set -eo pipefail
DATASET=$1
[ -n "$DATASET" ] || { echo "usage: qsub jobs/tune.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.experiment tune "$DATASET" ${EXTRA_LRS:+--lrs $EXTRA_LRS}
python3 -m b2j.experiment select "$DATASET"
date
