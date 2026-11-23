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
#$ -l h_rt=48:00:00
# LoRA fine-tuning ablation, one loss per job (tuning lr grid + edge rule, then seeds 1-5): qsub -N lora_<loss> jobs/finetune_lora.qsub.sh <loss>
set -eo pipefail
LOSS=$1
[ -n "$LOSS" ] || { echo "usage: qsub jobs/finetune_lora.qsub.sh <b2|soft_bce|hard_bce>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -m b2j.finetune endoscapes --loss "$LOSS" --workers 6
date
