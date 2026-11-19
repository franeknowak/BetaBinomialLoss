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
# Tuning grid + extension, selection and final runs of one dataset, with options passed to both stages, e.g.
# qsub -N tunefinal_endovit jobs/tune_final.qsub.sh endoscapes --cache cache/embeddings_endovit_mean --root results/endovit_mean_tuned
set -eo pipefail
[ -n "$1" ] || { echo "usage: qsub jobs/tune_final.qsub.sh <dataset> [--cache <dir>] [--root <dir>]"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.experiment tune "$@" --extend
python3 -m b2j.experiment final "$@"
date
