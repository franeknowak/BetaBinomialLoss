#$ -S /bin/bash
#$ -l tmem=8G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=8:00:00
#$ -N extract_sages
# Extracts the labelled SAGES frames (every 150th frame = one every 5 s at 30 fps = 0.2 fps) with the organisers'
# tools/preprocess_videos.py, unchanged, into datasets/sages/derived/<split>/frames/<video>/ (+ frame_map.csv).
# Usage: qsub jobs/extract_sages.qsub.sh train|test jpg|png
set -eo pipefail
SPLIT=${1:?usage: extract_sages.qsub.sh train|test jpg|png}
EXT=${2:?usage: extract_sages.qsub.sh train|test jpg|png}
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
DS=/cluster/project7/FranekLapChole/datasets/sages
python3 "$DS/tools/preprocess_videos.py" --dataset-root "$DS" --split "$SPLIT" --fps 0.2 --image-extension "$EXT" \
    | grep -v '^Extracted '
echo "$SPLIT: $(ls "$DS/derived/$SPLIT/frames" | wc -l) video folders, $(find "$DS/derived/$SPLIT/frames" -name "frame_*.$EXT" | wc -l) frames"
du -sh "$DS/derived/$SPLIT"
date
