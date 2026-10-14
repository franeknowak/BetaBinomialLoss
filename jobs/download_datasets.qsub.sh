#$ -S /bin/bash
#$ -l h_rt=4:00:00
#$ -l tmem=8G
#$ -wd /cluster/project7/FranekLapChole/datasets
#$ -j y
#$ -N dl_cifar10h_mhist
#$ -R y
# CPU-only download of CIFAR-10H and MHIST into /cluster/project7/FranekLapChole/datasets/{cifar10h,mhist}.
# Layout matches b2j/data.py (cifar10h_labels, mhist_labels).
#
# MHIST links are time-limited and must never be committed. Before submitting, write them to
# $MHIST_URLS (default datasets/mhist/.urls, chmod 600), one per line, in this order:
#   MD5SUMs.txt, annotations.csv, images.zip
# The file is deleted at the end of the job. Submit: qsub jobs/download_datasets.qsub.sh
set -euo pipefail
hostname; date
ROOT=/cluster/project7/FranekLapChole
DS=$ROOT/datasets
MHIST_URLS=${MHIST_URLS:-$DS/mhist/.urls}
CURL="curl -f -L --retry 5 --retry-delay 15 -sS"

# ------------------------------------------------------------------ CIFAR-10H
mkdir -p "$DS/cifar10h/images"
cd "$DS/cifar10h"
$CURL -o cifar10h-counts.npy https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy
$CURL -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo "c58f30108f718f92721af3b95e74349a  cifar-10-python.tar.gz" | md5sum -c -
tar -xzf cifar-10-python.tar.gz cifar-10-batches-py/test_batch
set +u   # site source files reference unset variables
source /share/apps/source_files/python/python-3.11.9.source
source "$ROOT/BetaBinomialLoss/dino_env/bin/activate"
set -u
python3 - <<'EOF'
# CIFAR-10 test set in its original order (= CIFAR-10H order) -> images/{id:05d}.png + test_labels.npy
import pickle, numpy as np
from PIL import Image
d = pickle.load(open("cifar-10-batches-py/test_batch", "rb"), encoding="bytes")
x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
y = np.array(d[b"labels"], dtype=np.int64)
for i, im in enumerate(x):
    Image.fromarray(im).save(f"images/{i:05d}.png")
np.save("test_labels.npy", y)
c = np.load("cifar10h-counts.npy")
assert c.shape == (10000, 10) == (len(y), 10), c.shape
n = c.sum(1)
print("cifar10h: n per image", n.min(), "-", n.max(), "| human-majority == CIFAR label:", (c.argmax(1) == y).mean())
EOF
rm -rf cifar-10-python.tar.gz cifar-10-batches-py
echo "CIFAR-10H: $(ls images | wc -l) images"

# ------------------------------------------------------------------ MHIST
mkdir -p "$DS/mhist"
cd "$DS/mhist"
[ -f "$MHIST_URLS" ] || { echo "missing $MHIST_URLS"; exit 1; }
mapfile -t U < "$MHIST_URLS"
[ "${#U[@]}" -ge 3 ] || { echo "need 3 URLs in $MHIST_URLS"; exit 1; }
dl() { local u="${1/dl=0/dl=1}"; $CURL -o "$2" "$u"; }   # Dropbox: force direct download
dl "${U[0]}" MD5SUMs.txt
dl "${U[1]}" annotations.csv
dl "${U[2]}" images.zip
rm -f "$MHIST_URLS"
cat MD5SUMs.txt
md5sum -c --ignore-missing MD5SUMs.txt
unzip -q -o images.zip
rm images.zip
head -3 annotations.csv
echo "MHIST: $(find . -name '*.png' | wc -l) png files"
date
