from pathlib import Path
import re
import mne

ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Chiso")

edf_files = sorted(ROOT.rglob("*.edf"))

print("num edf:", len(edf_files))

for p in edf_files[:10]:
    print("\n---", p.relative_to(ROOT), "---")
    try:
        raw = mne.io.read_raw_edf(str(p), preload=False, verbose=False)
        print("sfreq:", raw.info["sfreq"])
        print("n_channels:", len(raw.ch_names))
        print("duration_sec:", raw.n_times / raw.info["sfreq"])
        print("ch_names_first20:", raw.ch_names[:20])

        anns = raw.annotations
        print("annotations:", len(anns))
        if len(anns) > 0:
            for i in range(min(10, len(anns))):
                print(
                    i,
                    "onset=", anns.onset[i],
                    "duration=", anns.duration[i],
                    "desc=", anns.description[i],
                )
    except Exception as e:
        print("ERROR:", repr(e))
