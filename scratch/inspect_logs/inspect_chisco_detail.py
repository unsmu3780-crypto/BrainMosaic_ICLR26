from pathlib import Path
import pandas as pd
import re

ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Chiso")

print("ROOT:", ROOT)

print("\n===== textdataset xlsx samples =====")
xlsx_files = sorted((ROOT / "textdataset").glob("split_data_*.xlsx"))

for p in xlsx_files[:10]:
    df = pd.read_excel(p)
    print("\n---", p.name, "shape", df.shape, "---")
    print(df.head(10).to_string(index=False))

print("\n===== textdataset id list =====")
ids = []
for p in xlsx_files:
    m = re.search(r"split_data_(\d+)\.xlsx", p.name)
    if m:
        ids.append(int(m.group(1)))
print("num xlsx:", len(ids))
print("ids:", sorted(ids))

print("\n===== edf file pattern samples =====")
edf_files = sorted(ROOT.rglob("*.edf"))
print("num edf:", len(edf_files))
for p in edf_files[:80]:
    print(p.relative_to(ROOT))

print("\n===== edf run ids =====")
run_ids = []
for p in edf_files:
    m = re.search(r"run-0*(\d+)_eeg\.edf", p.name)
    if m:
        run_ids.append(int(m.group(1)))
print("num run ids:", len(run_ids))
print("min/max:", min(run_ids), max(run_ids))
print("unique first 80:", sorted(set(run_ids))[:80])
