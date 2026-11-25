#$ -S /bin/bash
#$ -l tmem=16G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=4:00:00
#$ -N inspect_sages
# Inspect the SAGES labels against the extracted frames (run after extract_sages for train and test).
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
export MPLBACKEND=Agg
python3 -m b2j.inspect_sages /cluster/project7/FranekLapChole/datasets/sages results/sages_inspect
date
