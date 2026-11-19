# Reconstruction evidence

Only `inputs.json`, `neural_inputs.npz`, and the numerical helpers supplied in this directory were used. No network, external labels, other directory data, trained encoder, or pretrained language decoder was used.

Reproduce the numerical analysis and the explicitly recorded human composition with:

```sh
OPENBLAS_NUM_THREADS=1 /Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python run_analysis.py
```

`final_predictions.json` is the interpreted answer. `reconstruction_evidence.json` records each decision, its uncertainties, a forced alignment of that interpretation, competing candidate scores, and overlapping calibration phrase matches. A forced alignment assesses a supplied candidate; it does not independently establish that candidate's words.

The independent, unrestricted numerical sequences are saved in `algorithm_only_predictions.json`, `consensus_algorithm_only.json`, `phrase_algorithm_only.json`, and `word_dtw_algorithm_only.json`. These are distinct from the final interpreted answer. `compare_candidates.py` contains the human proposed alternatives; its candidate rankings are conditional on those alternatives and are not an independent vocabulary-wide search.

## Numerical procedure

1. Fit a PCA to concatenated calibration frames only. Center all recordings with that calibration mean. Remove components 1 and 2, smooth with Gaussian sigma 1.2 frames, and normalize frames. Retrieve intervals using cosine similarity of resampled trajectories. A secondary comparison retains PCs 3–32, without replacing the main representation or selecting parameters using query labels.
2. Retrieve overlapping query windows against all calibration trajectories using slope-constrained subsequence DTW. Allowed moves are (1,1), (2,1), and (1,2), with a 0.035 slope penalty. `evidence_dtw.py` vectorizes the supplied recurrence and retains paths for leading matches. Detailed results are in `local_dtw_drop2.json`; readable summaries are in `dtw_evidence.txt` and `retrieval_evidence.txt`.
3. Infer calibration word intervals using known calibration transcript order and leave-one-recording-out exemplar similarities. Initial durations come from robust, bounded least squares on calibration sentence durations and word occurrence counts. Three fixed segmentation passes use only cosine between observed, resampled intervals plus a duration penalty. No neural feature weights or encoder are fitted. Each word is represented by its observed calibration intervals, not external prototypes. When a word has no other recording, neural evidence cannot identify its internal boundaries: a neutral score and a duration range are used. Excluded recordings are removed before calculating top-three support, including for rare words.
4. Independently assemble unrestricted query word sequences with interval dynamic programming, unrestricted counts, and no language model. For interval length L and duration expectation d, its contribution is L times cosine support, minus 0.13*d*log(L/d)^2, minus the declared span cost. The exemplar method combines the maximum and mean of the best three available occurrences equally. Consensus retrieval uses the normalized arithmetic mean of observed interval vectors. Phrase mosaic retrieval uses contiguous observed calibration ngrams, resampled to ten points, and its declared duration/span penalties. Word DTW compares each candidate interval to complete observed calibration word intervals with the same slope-constrained recurrence.
5. Inspect overlapping phrase evidence and the competing interval assignments. The final words and interpretive decisions are explicitly recorded in `finalize_reconstruction.py`. Some readings differ from unrestricted numerical output; the evidence records these changes and alternatives. Unusual phrases were retained where no better supported composition was available. No whole calibration sentence is used as a final query answer.

## Scope and limits

Word intervals are inferred, not annotated ground truth. Repeated phrases sometimes localize a word more reliably than isolated word matching; isolated matching sometimes confuses pronouns, auxiliaries, and adjacent words. Word counts are sensitive to duration and span costs. Scores are similarities, not calibrated probabilities. Singleton words lack an independent same-word neural example. The 2-frame initial and 3-frame final trimming is a working alignment convention, not an observed speech boundary.

All 16 query IDs are covered; final tokens are lowercase calibration-vocabulary words; every final token sequence differs from every complete calibration transcript. These are structural checks, not accuracy evaluation. No evaluation target was available. The least certain overall readings are `tda989373`, `t4f16b000`, and `t62254eda`.
