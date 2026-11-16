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
#$ -l h_rt=6:00:00
# Rater-bias variants of B² and soft BCE (datasets with rater identities): qsub -N rb_<dataset> jobs/rater_bias.qsub.sh <endoscapes|amibr>
# Extra options are passed on, e.g. amibr --seeds 1
set -eo pipefail
DS=$1
[ -n "$DS" ] || { echo "usage: qsub jobs/rater_bias.qsub.sh <endoscapes|amibr>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.rater_bias "$@"
date
