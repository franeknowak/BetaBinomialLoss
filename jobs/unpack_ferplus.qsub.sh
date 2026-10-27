#$ -S /bin/bash
#$ -l tmem=8G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=2:00:00
#$ -N unpack_ferplus
# Unpack the FER+ Kaggle zip next to the labels, check every member arrived, check the loader finds every indexed
# face, then delete the zip (its md5 stays in SOURCE_MD5.txt).
set -eo pipefail
hostname; date
DS=/cluster/project7/FranekLapChole/datasets/ferplus
cd "$DS"
ZIP=archive.zip
[ -f "$ZIP" ] || { echo "no $ZIP in $DS"; exit 1; }
quota -s | grep -A1 FranekLapChole || true

unzip -tq "$ZIP"                                              # CRC check before anything is written
unzip -Z1 "$ZIP" | grep -v '/$' > /tmp/ferplus_members_$$.txt
echo "members in zip: $(wc -l < /tmp/ferplus_members_$$.txt)"
unzip -q -n "$ZIP"
MISSING=0
while IFS= read -r f; do [ -f "$f" ] || MISSING=$((MISSING + 1)); done < /tmp/ferplus_members_$$.txt
rm -f /tmp/ferplus_members_$$.txt
echo "members missing after unpacking: $MISSING"
[ "$MISSING" -eq 0 ] || { echo "unpacking incomplete; zip kept"; exit 1; }

cd /cluster/project7/FranekLapChole/BetaBinomialLoss
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
python3 - <<'EOF'
import sys
import numpy as np
from b2j import data as D
root = D.DATA_ROOT / "ferplus"
idx = __import__("pandas").read_csv(root / "zip_index.csv", dtype=str)
files = [root / m for m in idx["member"]]
bad = [str(f) for f in files if not f.exists()]
print("indexed faces:", len(files), "| without a file in the unpacked copy:", len(bad), bad[:3])
same = all(np.array_equal(np.asarray(D.open_image(str(f))), np.asarray(D.open_image(f"{root / a}{D.ZIP_SEP}{m}")))
           for f, a, m in list(zip(files, idx["archive"], idx["member"]))[:: max(1, len(files) // 200)])
print("sampled unpacked images identical to the zip copies:", same)
sys.exit(0 if not bad and same else 1)
EOF

cd "$DS"
rm "$ZIP"
echo "deleted $ZIP"
cd /cluster/project7/FranekLapChole/BetaBinomialLoss
python3 -c "
from b2j import data as D
t = D.load_table('ferplus')
print('loader now reads the folder:', D.ZIP_SEP not in t.paths[0], t.paths[0])
print('faces:', len(t.ids), {s: int((t.split == s).sum()) for s in ('train', 'val', 'test')})
"
cd "$DS"
ls
quota -s | grep -A1 FranekLapChole || true
date
