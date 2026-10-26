#$ -S /bin/bash
#$ -l h_rt=4:00:00
#$ -l tmem=8G
#$ -wd /cluster/project7/FranekLapChole/datasets
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -N dl_datasets
#$ -R y
# CPU-only download into /cluster/project7/FranekLapChole/datasets/<name> (= b2j.data.DATA_ROOT / <name>).
# Usage: qsub jobs/download_datasets.qsub.sh cifar10h
#        qsub jobs/download_datasets.qsub.sh mhist
#        qsub jobs/download_datasets.qsub.sh amibr
#        qsub jobs/download_datasets.qsub.sh ferplus   (fer2013new.csv + Kaggle zip copied into datasets/ferplus first)
# MHIST links are time-limited and never committed: write them to datasets/mhist/.urls (chmod 600),
# one per line: MD5SUMs.txt, annotations.csv, images.zip. The job deletes the file.
set -euo pipefail
hostname; date
WHAT=${1:?usage: qsub download_datasets.qsub.sh cifar10h|mhist|amibr|ferplus}
ROOT=/cluster/project7/FranekLapChole
DS=$ROOT/datasets
CURL="curl -f -L --retry 5 --retry-delay 15 -sS"

if [ "$WHAT" = cifar10h ]; then
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

elif [ "$WHAT" = mhist ]; then
    cd "$DS/mhist"
    URLS=.urls
    if [ -f MD5SUMs.txt ] && [ -f annotations.csv ] && [ -f images.zip ]; then
        echo "files already downloaded; skipping download"
    else
        [ -f "$URLS" ] || { echo "missing $DS/mhist/$URLS"; exit 1; }
        mapfile -t U < "$URLS"
        rm -f "$URLS"
        [ "${#U[@]}" -ge 3 ] || { echo "need 3 URLs"; exit 1; }
        dl() { local u="${1/dl=0/dl=1}"; $CURL -o "$2" "$u"; }   # Dropbox: force direct download
        dl "${U[0]}" MD5SUMs.txt
        dl "${U[1]}" annotations.csv
        dl "${U[2]}" images.zip
    fi
    cat MD5SUMs.txt
    md5sum -c MD5SUMs.txt   # cluster coreutils has no --ignore-missing; the file lists images.zip only
    unzip -q -o images.zip
    rm images.zip
    head -3 annotations.csv
    echo "MHIST: $(find . -name '*.png' | wc -l) png files"
elif [ "$WHAT" = amibr ]; then
    # AMi-Br (Bertram et al., BVM 2025): CSV with 3 experts' atypical votes + 128 px patches, public GitHub repo.
    mkdir -p "$DS/amibr"
    cd "$DS/amibr"
    rm -rf repo
    git clone -q --depth 1 https://github.com/DeepMicroscopy/AMi-Br.git repo
    (cd repo && git log -1 --format='AMi-Br commit %H %cd') | tee SOURCE_COMMIT.txt
    mv repo/AMI-BR.csv repo/README.md .
    rm -rf patches && mv repo/patches .
    rm -rf repo
    head -3 AMI-BR.csv
    echo "AMi-Br: $(tail -n +2 AMI-BR.csv | wc -l) rows, $(find patches -name '*.png' | wc -l) png files"
elif [ "$WHAT" = ferplus ]; then
    # The user copies fer2013new.csv (Microsoft FERPlus repo) and the Kaggle zip (arnabkumarroy02/ferplus) here
    # first (Kaggle needs a login). Nothing is unpacked: b2j.prepare_ferplus writes zip_index.csv (images read in place).
    cd "$DS/ferplus"
    ZIP=$(ls *.zip | head -1)
    [ -f fer2013new.csv ] && [ -n "$ZIP" ] || { echo "need fer2013new.csv and the Kaggle zip in $DS/ferplus"; exit 1; }
    md5sum "$ZIP" fer2013new.csv | tee SOURCE_MD5.txt
    set +u
    source /share/apps/source_files/python/python-3.11.9.source
    source "$ROOT/BetaBinomialLoss/dino_env/bin/activate"
    set -u
    PYTHONPATH="$ROOT/BetaBinomialLoss" python3 -m b2j.prepare_ferplus "$ZIP" "$DS/ferplus"
    wc -l zip_index.csv
else
    echo "unknown dataset $WHAT"; exit 1
fi
date
