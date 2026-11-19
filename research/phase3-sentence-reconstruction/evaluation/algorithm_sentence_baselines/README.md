# Algorithm-only sentence contribution comparison

The primary question is how much final interpretation/composition Sol adds beyond its numerical processing. All baseline code/results are independent of Sol inputs. No fitted word decoder, prototype, prediction, or baseline output was copied into a Sol sandbox. No model calls were made for this comparison.

## Results

| Evaluation | Algorithm | Sol | Net extra errors removed by Sol |
|---|---:|---:|---:|
| Development 8: frozen full-evidence replay | 14/39 word errors, 35.90% WER; 1 exact | 10/39, 25.64%; 2 exact | 4 |
| Fresh 16: fixed generic workflow | 34/71, 47.89%; 0 exact | 29/71, 40.85%; 4 exact | 5 |
| Fresh strictly novel whole-target subset 14 | 31/64, 48.44%; 0 exact | 27/64, 42.19%; 3 exact | 4 |

The generic result uses a calibration-locked rule, regenerates neural evidence without Sol, and reads only the fresh case's inputs.json and neural_inputs.npz at application. It does not read Sol helpers, evidence, hypotheses, or predictions. Its numerical algorithm inventory was discovered during development, so it still inherits that research discovery. The fresh set is one block in the same participant/session; this is not population-level evidence or a claim of irreducible intelligence.

The paired WER improvement ranges from descriptive within-block utterance resampling include zero: development −2.70 to +20.00 percentage points; fresh16 −6.41 to +21.25. These ranges do not model within-block/session correlation. The observed additional Sol contribution is modest in edit count. A stronger algorithmic assembler might close it.

## A. Frozen Sol-orchestrated evidence replay

`full_evidence/` exposes all twelve structured query-evidence families to calibration-only selection: five global-DTW representations, five local cosine/resampling representations, subsequence-DTW, and edge/prefix retrieval. It also provides the actual calibration-to-calibration inspection and prefix contexts as optional automatic boundary refinement. The query-specific prefix comparison subset duplicates the corresponding complete edge scores.

No final predictions or hypothesis prose from inference_evidence.json, reconstruction.json, or model commentary are used. Diagnostic eigenvalue/norm/PC printouts are recorded in the inventory as diagnostics rather than direct word-score families. The computed-but-unused smoothed PCA matrix never generated a distinct retrieval family in the actual call.

This replay inherits Sol's choice of numerical algorithms and particular calibration context IDs. It is final assembly given Sol-orchestrated evidence, not a causal removal of every Sol contribution. All original structured scores/paths are available to model selection; they are not all forced to receive positive weight.

Three-fold CV holds out calibration blocks 1, 4, and 6. PCA, retrieval candidates, calibration-context refinement, and text priors are regenerated using the other two blocks. Among 576 family/ensemble and assembly settings, CV selects the central drop2 subsequence-DTW family alone, top12 candidates, uniform weak calibration word timing, no context refinement, no language prior, and a −2.5 word-transition penalty. Consequently the expanded replay reproduces the earlier central-evidence result rather than introducing a query-selected improvement.

The C99 DP helper exactly reproduces all 2,652 saved development subsequence-DTW matches, IDs, and paths, with zero score difference. Prefix path recovery reproduces all 3,840 saved edge scores with zero difference. Calibration prefix contexts use the actually logged twelve-frame comparison, while query edge contexts include 5/8/12/16 frames.

## B. Fixed generic workflow

The generic workflow generates the same predetermined numerical inventory from neural arrays, with evenly spaced calibration context IDs instead of Sol-selected IDs. Its own calibration-only CV again selects the central subsequence-DTW assembler. No query labels, word counts, or word boundaries are used for selecting parameters or making predictions. Target text is read only by the separate final evaluator after the predictions have been saved.

Word timing is weakly initialized from calibration transcript word counts, either uniform or proportional to word spelling length, and selected by CV. It is not an observed word alignment. Retrieved calibration paths project these weak labels to query frames. Weighted votes feed an unconstrained word-sequence Viterbi decoder, with optional calibration-only bigram priors and duration regularization. It can generate utterances absent from calibration; it is not limited to copying known whole sentences.

The first uncached fresh16 application took 12.73 seconds, including all numerical evidence families, versus 0.43 seconds to assemble the already-saved development evidence. These are different cost scopes. CV timing in configuration files can include cache hits; it is not a clean cold-start benchmark. Optimized C numerical code also differs from the exploratory Python code authored during Sol inference.

## Other references and limits

Nearest whole-recording drop2 DTW returns 66.67% development WER and zero exact sentences. Because targets are absent from calibration, copying alone is not a sufficient sentence-composition comparator.

A calibration-only text-splice candidate generator was also tested, reproducing the 2,340 unseen candidates from the no-signal prior script. Rebuilding it within CV folds gives only 9/120 held-out calibration target coverage; none of the eight development targets is in the full bank. Its 43.59% development WER / zero exact results are therefore a restrictive-hypothesis reference, not proof of a compositional Sol advantage.

A separate trained weak-word emission/forced-alignment baseline was calibrated (66.27% calibration-CV WER) but not predicted or scored after the full-pipeline ablation took priority. It is secondary, independent, and not part of either primary result. No learned word weights were supplied to Sol.

## Reproduce

From the project root, using the existing scientific environment:

```sh
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_algorithm_comparison.py calibrate
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_algorithm_comparison.py calibrate_splice
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_algorithm_comparison.py predict
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_algorithm_comparison.py evaluate

work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py calibrate
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py predict
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py evaluate

work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py calibrate_generic
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py predict_generic --case sentences_t12_fresh
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py evaluate_generic --case sentences_t12_fresh
```

`full_evidence/contribution_summary.json` contains the paired results and strict novelty subset. Configuration locks, locked predictions, scored rows, full CV scores, neural retrieval caches, source hashes, and the exact generic-selector source at its recorded lock are retained. The generic source snapshot verifies its historical recorded SHA256. Later changes add participant-specific calibration, tagged predictions, and dynamic vocabulary dimensions; the T12 transformations and selected parameters are unchanged. The pre-vocabulary-generalization source is also preserved.


## T16 extension (development and blinded confirmation)

The T12 selected parameters were transferred unchanged to T16. The lexicon is always derived from the supplied calibration transcripts: T16 has49 words (goodbye is absent), not T12's50. This required generalizing an array dimension, not retuning decoding parameters. Inputs contain80 calibration sentences from blocks1/4 and64 i6v TX plus64 power channels.

A separate two-fold calibration-only selection holds block1 or4 out, recomputing PCA, all numerical evidence, context refinement and text priors on the other block. It selects local whitening/cosine, top12, uniform initial word timing, automatic calibration-context refinement, LM0.5 and transition penalty−2.5 (39.04% calibration-CV WER).

Known development8, scored only from the already available development evaluation JSON: transferred rule12/37 errors (32.43%,1 exact), T16-calibrated rule13/37 (35.14%,0 exact), Sol3/37 (8.11%,5 exact). Seven targets are absent even as calibration spans; all five Sol exact outputs belong to that strict subset. This is exploratory because ROI selection used an earlier diagnostic on the same eight recordings.

Both algorithm versions have saved predictions on the32 reserved recordings without reading their targets. `t16_confirmation_baseline_freeze.json` records prediction/config/code hashes; root owns evaluation after the freezes. No confirmation target text was loaded by this agent.

```sh
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py predict_generic --case sentences_t16_confirm
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py calibrate_case --case sentences_t16_fresh_dev
work/venv/bin/python outputs/neural_language_reconstruction/code/sentence_full_evidence_comparison.py predict_generic --case sentences_t16_confirm --configuration generic_sentences_t16_fresh_dev_configuration_lock.json --tag t16cal
```

Targets for final confirmation scoring must remain separate from development. The evaluator action loads source sentenceText; do not run it until root authorizes unblinding. To score known development without loading the source's reserved labels, use only its already-scored8 evaluation JSON as in `t16_development_comparison.json`.

The T16 method audit is `../t16_agentic_method_audit.json`. It finds no target/network/pretrained-decoder reads, but documents that Sol constructed69 labeled calibration interval exemplars across17 words and used numerical word-candidate scores. Those approximate intervals were inferred by Sol; raw source references to human interpretation are misnomers. This is not exclusively unlabeled retrieval, and no complete algorithm-only sentence was assembled by the model-authored numerical scripts. Raw model artifacts were preserved.
