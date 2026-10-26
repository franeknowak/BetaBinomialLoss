#$ -S /bin/bash
#$ -l tmem=32G
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=8:00:00
#$ -N gpu_job
#$ -t 1-4
# Template for a GPU job array (copy, rename -N, replace the command). Each task runs its share of the work items:
# b2j.cluster.task_slice(items, $SGE_TASK_ID, 4). Drop the -t line for a single job.
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.8.source
export HF_HOME=/cluster/project7/FranekLapChole/BetaBinomialLoss/.hf_cache
export HF_HUB_OFFLINE=1
source ./dino_env/bin/activate
git log --oneline -1
python3 -m b2j.cluster preflight
# python3 -m <module> --task $SGE_TASK_ID --ntasks 4
date
