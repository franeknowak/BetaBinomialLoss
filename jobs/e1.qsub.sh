#$ -S /bin/bash
#$ -l tmem=8G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=1:00:00
#$ -N e1
# Phase 2, E1 (cut-off fitted on validation) for one dataset: qsub -N e1_<dataset> jobs/e1.qsub.sh <dataset>
set -eo pipefail
[ -n "$1" ] || { echo "usage: qsub jobs/e1.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
export MPLBACKEND=Agg
python3 -m b2j.e1 "$1"
date
