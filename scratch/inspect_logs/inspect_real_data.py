from pathlib import Path
from collections import Counter

DATA_ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet")

datasets = ["ChineseEEG-2", "Chiso", "Zuco-1", "Zuco-2"]

for ds in datasets:
    root = DATA_ROOT / ds
    print("\n" + "=" * 80)
    print(ds)
    print("path:", root)
    print("exists:", root.exists())

    exts = Counter()
    files = []

    for p in root.rglob("*"):
        if p.is_file():
            exts[p.suffix.lower() or "<no_ext>"] += 1
            files.append(p)

    print("\n[extensions]")
    for ext, n in exts.most_common(30):
        print(ext, n)

    print("\n[largest files]")
    for p in sorted(files, key=lambda x: x.stat().st_size, reverse=True)[:30]:
        print(f"{p.stat().st_size / 1024 / 1024:9.2f} MB  {p}")

    print("\n[candidate metadata files]")
    for p in files:
        name = p.name.lower()
        if p.suffix.lower() in [".csv", ".tsv", ".xlsx", ".xls", ".json", ".mat", ".pt", ".pth"] or any(
            k in name for k in ["sentence", "text", "label", "word", "stimuli", "material", "metadata"]
        ):
            print(p)
