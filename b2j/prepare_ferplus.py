"""FER+ images from the Kaggle copy arnabkumarroy02/ferplus (v3: 112 px, class folders, augmented training set).

  python -m b2j.prepare_ferplus <zip> [<dataset dir>]     (cluster job: jobs/download_datasets.qsub.sh ferplus)

Nothing is unpacked (cluster file limit, user 2026-10-19). Reports what the zip holds, then writes
<dataset dir>/zip_index.csv (id, archive, member) for the originals named in fer2013new.csv (the faces kept by
b2j.data.ferplus_labels); b2j.data reads them inside the zip. Augmented copies are ignored. A name that maps to
several different files in the zip is ambiguous and left out.
"""
import collections
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from b2j.data import DATA_ROOT, ferplus_labels


def main(zip_path, root=DATA_ROOT / "ferplus"):
    root, zip_path = Path(root), Path(zip_path)
    (root / "zip_index.csv").unlink(missing_ok=True)          # label the full csv, not a previous index
    t = ferplus_labels(root)
    want = dict(zip(t.ids, t.split))
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if not n.endswith("/")]
    by, other = collections.defaultdict(list), []
    for n in names:
        b = n.rsplit("/", 1)[-1]
        (by[b[:-4]] if re.fullmatch(r"fer\d{7}\.png", b) else other).append(n)
    print(f"zip: {len(names)} files, {sum(map(len, by.values()))} with FER+ names ({len(by)} distinct), "
          f"{len(other)} others, e.g. {other[:5]}")
    conflict = {b for b, ns in by.items() if len(ns) > 1 and len({hashlib.md5(z.read(n)).hexdigest() for n in ns}) > 1}
    print(f"names with several different files: {len(conflict)}, e.g. {[by[b] for b in list(conflict)[:3]]}")
    print("top-level folders of FER+-named files:", dict(collections.Counter(n.split("/")[0] for ns in by.values() for n in ns)))
    ok = [i for i in want if i in by and i not in conflict]
    for s in ("train", "val", "test"):
        ids = [i for i, x in want.items() if x == s]
        have = sum(i in by and i not in conflict for i in ids)
        print(f"{s}: {have} / {len(ids)} kept faces present ({have / len(ids):.1%})")
    from PIL import Image
    sizes = collections.Counter()
    for i in ok[:500]:
        with Image.open(io.BytesIO(z.read(by[i][0]))) as im:
            sizes[(im.size, im.mode)] += 1
    print(f"sizes/modes of the first 500: {dict(sizes)}")
    pd.DataFrame({"id": ok, "archive": zip_path.name, "member": [by[i][0] for i in ok]}).to_csv(
        root / "zip_index.csv", index=False)
    print(f"wrote {root / 'zip_index.csv'}: {len(ok)} faces")


if __name__ == "__main__":
    main(*sys.argv[1:])
