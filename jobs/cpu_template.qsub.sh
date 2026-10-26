#$ -S /bin/bash
#$ -l tmem=16G
# arbuckle: Illegal instruction (job 7575103)
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=4:00:00
#$ -N cpu_job
# Template for a CPU job (copy, rename -N, replace the command). Chain after another job with
# qsub -hold_jid <job id> jobs/<name>.qsub.sh
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
export MPLBACKEND=Agg
# python3 -m <module>
date
