#$ -S /bin/bash
#$ -l tmem=8G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=1:00:00
#$ -N e3
# Phase 2, E3 (setting aside uncertain frames) for one dataset: qsub -N e3_<dataset> jobs/e3.qsub.sh <dataset>
set -eo pipefail
[ -n "$1" ] || { echo "usage: qsub jobs/e3.qsub.sh <dataset>"; exit 1; }
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
export MPLBACKEND=Agg
python3 -m b2j.e3 "$1"
date
