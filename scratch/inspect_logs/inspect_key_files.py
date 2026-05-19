from pathlib import Path
import json
import pandas as pd
import scipy.io as sio
import zipfile

ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet")

def show(path):
    path = Path(path)
    print(f"\n--- {path} ---")
    print("exists:", path.exists())
    if path.exists() and path.is_file():
        print("size_MB:", round(path.stat().st_size / 1024 / 1024, 2))

def inspect_table(path, n=5):
    path = Path(path)
    show(path)
    if not path.exists():
        return
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(path)
            print("sheets:", xls.sheet_names)
            for sh in xls.sheet_names[:2]:
                df = pd.read_excel(path, sheet_name=sh)
                print("sheet:", sh, "shape:", df.shape)
                print("columns:", list(df.columns))
                print(df.head(n).to_string())
        elif path.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(path, sep="\t" if path.suffix.lower()==".tsv" else ",")
            print("shape:", df.shape)
            print("columns:", list(df.columns))
            print(df.head(n).to_string())
        elif path.suffix.lower() in [".json", ".jsonl"]:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            print(txt[:2000])
    except Exception as e:
        print("ERROR:", repr(e))

def inspect_mat(path):
    path = Path(path)
    show(path)
    if not path.exists():
        return
    try:
        data = sio.loadmat(path, simplify_cells=True)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print("mat keys:", keys[:30])
        for k in keys[:10]:
            v = data[k]
            print(" ", k, "type:", type(v), "shape:", getattr(v, "shape", None))
            if isinstance(v, dict):
                print("   dict keys:", list(v.keys())[:30])
    except NotImplementedError as e:
        print("MAT v7.3/HDF5, need h5py:", repr(e))
    except Exception as e:
        print("ERROR:", repr(e))

def inspect_zip(path):
    path = Path(path)
    show(path)
    if not path.exists():
        return
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            print("zip items:", len(names))
            for name in names[:50]:
                print(" ", name)
    except Exception as e:
        print("ERROR:", repr(e))

print("ROOT:", ROOT)

# Chisco text metadata
print("\n================ Chisco key files ================")
for p in sorted((ROOT / "Chiso" / "textdataset").glob("*.xlsx"))[:5]:
    inspect_table(p, n=3)

# ChineseEEG-2 metadata
print("\n================ ChineseEEG-2 key files ================")
for p in (ROOT / "ChineseEEG-2").rglob("*"):
    if p.is_file() and p.suffix.lower() in [".tsv", ".json", ".txt"] and any(k in p.name.lower() for k in ["participants", "events", "stim", "sentence", "material", "metadata", "task"]):
        inspect_table(p, n=3)

# Look inside one ChineseEEG zip only
zips = sorted((ROOT / "ChineseEEG-2").rglob("*.zip"))
if zips:
    inspect_zip(zips[0])

# ZuCo-1 small metadata files
print("\n================ ZuCo-1 key files ================")
for p in sorted((ROOT / "Zuco-1").rglob("*")):
    if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".json"] and p.stat().st_size < 20 * 1024 * 1024:
        inspect_table(p, n=3)

# ZuCo-2 metadata
print("\n================ ZuCo-2 key files ================")
for p in sorted((ROOT / "Zuco-2").rglob("*")):
    if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".json", ".jsonl"] and p.stat().st_size < 20 * 1024 * 1024:
        inspect_table(p, n=3)

# One preprocessed mat from ZuCo-1/2
for dataset in ["Zuco-1", "Zuco-2"]:
    mats = sorted((ROOT / dataset).rglob("*Preprocessed*/*EEG.mat"))
    if mats:
        print(f"\n================ {dataset} one preprocessed EEG mat ================")
        inspect_mat(mats[0])
