#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_zuco_sr.py
python scripts/prepare_zuco_nr.py
python scripts/prepare_zuco_tsr.py

python scripts/build_zuco_sr_text_assets_inputs.py
python scripts/build_zuco_nr_text_assets_inputs.py
python scripts/build_zuco_tsr_text_assets_inputs.py

echo "[OK] prepared ZuCoSR / ZuCoNR / ZuCoTSR unified splits and configs"
