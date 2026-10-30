#$ -S /bin/bash
#$ -l tmem=16G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=4:00:00
#$ -N explore
# CPU analyses of explore/: qsub -N <name> jobs/explore_cpu.qsub.sh <module> <args>
#   e.g. explore.run phase2 mhist | explore.run sub mhist 3 | explore.evalsub mhist | explore.bootstrap mhist --m 3
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
python3 -m "$@"
date
