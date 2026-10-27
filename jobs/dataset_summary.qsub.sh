#$ -S /bin/bash
#$ -l tmem=8G
#$ -l hostname=!arbuckle*
#$ -wd /cluster/project7/FranekLapChole/BetaBinomialLoss
#$ -o /cluster/project7/FranekLapChole/BetaBinomialLoss/jobs/logs/
#$ -j y
#$ -l h_rt=1:00:00
#$ -N dataset_summary
# Per dataset and split, as the loaders in b2j.data build them: images, clusters, raters per image, majority-vote
# prevalence per label and how often the panel is unanimous.
set -eo pipefail
hostname; date
source /share/apps/source_files/python/python-3.11.9.source
source ./dino_env/bin/activate
git log --oneline -1
python3 - <<'EOF'
import numpy as np
from b2j import data as D
for name in ("endoscapes", "cifar10h", "mhist", "amibr", "ferplus"):
    t = D.load_table(name)
    print(f"\n=== {name}: {len(t.ids)} images, {t.k.shape[1]} label(s), {len(np.unique(t.cluster))} clusters")
    for s in ("train", "val", "test"):
        m = t.split == s
        k, n = t.k[m], t.n[m]
        maj = (k > n / 2).mean(0)
        unan = ((k == 0) | (k == n)).mean(0)
        print(f"  {s:5s}: {m.sum():6d} images, {len(np.unique(t.cluster[m])):5d} clusters, raters/image "
              f"min {n.min():g} median {np.median(n):g} max {n.max():g}")
        print(f"         majority-positive %: {np.round(100 * maj, 1).tolist()}")
        print(f"         unanimous %:         {np.round(100 * unan, 1).tolist()}")
EOF
date
