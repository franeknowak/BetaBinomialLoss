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
#$ -l h_rt=4:00:00
# Endoscapes, one criterion: train on unanimous frames, val + test on split frames: qsub -N unan_<C> jobs/unanimous.qsub.sh <C1|C2|C3>
# Extra options are passed on, e.g. C1 --seeds 1
set -eo pipefail
CRIT=$1
[ -n "$CRIT" ] || { echo "usage: qsub jobs/unanimous.qsub.sh <C1|C2|C3>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.unanimous "$@"
date
