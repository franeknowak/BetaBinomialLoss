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
# Cache the DINOv3 [CLS] embeddings of one dataset: qsub -N embed_<dataset> jobs/embed.qsub.sh <dataset>
set -eo pipefail
DATASET=$1
[ -n "$DATASET" ] || { echo "usage: qsub jobs/embed.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.embed "$DATASET" --workers 4
date
