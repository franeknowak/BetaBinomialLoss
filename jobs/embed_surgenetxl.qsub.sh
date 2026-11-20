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
# SurgeNetXL (DINOv3 ViT-B/16, authors' architecture via transformers; see jobs/install_transformers.qsub.sh) embeddings of one dataset (downloads the HF weights first): qsub -N surgenetxl_<dataset> jobs/embed_surgenetxl.qsub.sh <dataset>
set -eo pipefail
DATASET=$1; shift || true
[ -n "$DATASET" ] || { echo "usage: qsub jobs/embed_surgenetxl.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=0   # the SurgeNetXL weights are fetched once into HF_HOME
source ./dino_env/bin/activate
git log --oneline -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -m b2j.cluster preflight
python3 -c "from huggingface_hub import hf_hub_download as d; print(d('rlpddejong/SurgeNetXL_DINOv1-v3', 'DINOv3_ViTb16_size336_SurgeNetXL.pth'))"
python3 -m b2j.embed "$DATASET" --workers 4 --encoder surgenetxl "$@"
date
