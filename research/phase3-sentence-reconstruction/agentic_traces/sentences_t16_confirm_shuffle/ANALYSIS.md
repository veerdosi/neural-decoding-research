All associations in this analysis use inputs.json and neural_inputs.npz. No network, outside data, decoder weights, evaluation targets, or agents were used.

The PCA was fitted to calibration rows only. Matching used Gaussian smoothing (sigma 1.2 bins), omission of the first two PCA components, row normalization, cosine interval comparison, and slope-constrained DTW. fast_dtw.c implements the same three DTW steps as the supplied helper, with an additional fixed-end option for comparing complete candidate intervals.

calibration_intervals.json contains analyst estimates of calibration word boundaries, inferred from overlapping calibration retrieval. These are approximate labels on intervals, not supplied boundaries. template_reconstruction.py uses them in an unrestricted, duration-constrained interval-template segmentation with a word penalty. refine_alignment.py performs one calibration-only leave-one-recording-out boundary refinement and a second template segmentation. dtw_vote.py transfers the approximate interval labels through the independently retrieved DTW paths. These numerical assembly outputs are saved separately and are not the final reconstruction:

- algorithm_only_predictions.json
- algorithm_only_refined_predictions.json
- algorithm_only_vote_predictions.json

The final interpretation is composed explicitly in compose_final.py, with a per-query explanation and approximate word spans in interpreted_evidence.json. It combines overlapping matches, resolves or retains conflicting candidates, and uses the supplied constraint that no entire query transcript occurs in calibration. It does not treat grammar as evidence. Some interpretations remain weak, as recorded per query. Similarity values are retrieval scores, not calibrated probabilities. A single calibration occurrence, for example success, cannot provide cross-instance confirmation.

Reproduce the main evidence with the provided Python executable and OPENBLAS_NUM_THREADS=1. Run these commands in this directory, in order:

```
export OPENBLAS_NUM_THREADS=1
PYTHON=/Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python
"$PYTHON" prepare_pca.py
"$PYTHON" local_retrieval.py drop2
cc -O3 -dynamiclib -o fast_dtw.dylib fast_dtw.c
"$PYTHON" fast_evidence.py
"$PYTHON" template_reconstruction.py
"$PYTHON" refine_alignment.py
"$PYTHON" dtw_vote.py
"$PYTHON" phrase_evidence.py
"$PYTHON" final_diagnostics.py
"$PYTHON" summarize_evidence.py
"$PYTHON" compose_final.py
```

Additional calibration checks are saved as inspect_drop2_*.json, calibration_inspection.txt and more_calibration_inspection.txt. They can be reproduced with inspect_intervals.py and the calibration IDs appearing in their filenames. Bulky retrieval scores and paths are retained in local_drop2.json, dtw_evidence.json, template_evidence.json and dtw_vote_evidence.json. Human-readable versions are in the corresponding *_readable.txt files. final_predictions.json is the final JSON response saved locally.
