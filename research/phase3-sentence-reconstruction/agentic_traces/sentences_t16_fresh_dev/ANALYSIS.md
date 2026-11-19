Reproduction (run in this directory)

All commands use the supplied Python executable and `OPENBLAS_NUM_THREADS=1`.

1. `python prepare_pca.py`
2. `python local_retrieval.py drop2`
3. `python edge_retrieval.py drop2 > edge_report.txt`
4. `python local_dtw.py drop2`
5. `python inspect_intervals.py drop2 c031 c039 c055 c018 c072 c066 c062 c045 c008 c041 c028 c014 c050 c035 c056 > calibration_overlap_report.txt`
6. `python inspect_intervals.py drop2 c021 c004 c070 c058 c003 c036 c034 c019 c063 c061 > additional_overlap_report.txt`
7. `python targeted_evidence.py > targeted_report.txt`
8. `python record_evidence.py`

The executable used was `/Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python`.

The supplied PCA is computed only from calibration frames. The retrieval helper
centers all trajectories with that mean, projects into that basis, removes the
first two components, smooths with sigma 1.2 rows, and normalizes each frame.
Interval retrieval resamples each interval to eight points. Subsequence DTW uses
cosine similarity, steps 1:1, 2:1, 1:2, and the supplied 0.035 slope penalty.
Only the drop2 configuration was run. No parameter sweep or model training was
performed.

`local_drop2.json`, `edge_drop2.json`, and `local_dtw_drop2.json` retain numerical
evidence. The two `inspect_drop2_*.json` files retain calibration overlap matches.
`targeted_evidence.json` retains targeted complete-interval comparisons, including
alternative words, with explicit manually interpreted exemplar boundaries.
`reconstruction_evidence.json` connects each interpreted word to numerical rows
and records uncertainty. `final_interpreted_predictions.json` contains the final
assistant composition. The numerical procedures do not assemble a new sentence,
so no algorithm-only decoded token sequence exists.

The strongest revisions from initial retrieval impressions were have rather
than like in te7fda903, and the additional not before hungry in te1c6177b. In the
targeted tests, query te7fda903 rows 17:26 match two have intervals with scores
0.262 and 0.248, versus the two best like intervals at 0.168 and 0.164. Query
te1c6177b rows 21:30 favor not with a top-two mean of 0.300. These comparisons
use approximate manually associated calibration intervals; they are supporting
evidence, not calibrated confidence estimates.

ta84f22ff remains uncertain: you, have, and here each have support, but an extra
it cannot be resolved confidently. The final interpretation omits it because
the have and here evidence windows cover the middle and ending, while it does
not recur consistently across its calibration examples. No word was inserted
solely to improve grammar. The final words may therefore sound awkward.
