#!/bin/zsh
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
PY=/Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python
cc -O3 -shared -fPIC fast_dtw.c -o fast_dtw.so
$PY prepare_pca.py
$PY local_retrieval.py drop2
$PY fast_evidence.py
$PY suffix_evidence.py > suffix_evidence.txt
$PY interval_assembly.py
$PY inspect_intervals.py drop2 c016 c028 c091 c062 c105 c066 c003 > calibration_interval_evidence.txt
$PY refine_evidence.py
$PY phrase_assembly.py
$PY candidate_comparisons.py
$PY calibration_feature_check.py
$PY save_reconstruction.py
