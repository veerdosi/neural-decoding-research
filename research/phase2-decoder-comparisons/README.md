# Phase 2 reproducibility bundle

Actual model outputs are in `responses/`; scored predictions are in `evaluation/`. Every retained model condition uses `gpt-6.1-sol` through authenticated Codex with high reasoning effort. Model inputs are in `prompts/`. Conventional decoders are independent comparisons and never supply predictions or weights to Sol.

## Inspect or rescore the existing evidence

Install the packages in `requirements.txt` in a Python environment, then run from a project directory:

```sh
python /absolute/path/neural_decoding_phase2/code/analyze.py
python /absolute/path/neural_decoding_phase2/code/figures.py
```

This reads saved responses, verifies IDs and labels, checks for model tool use, and recreates statistics and figures. The included feature arrays make offline rescoring possible. The original large MATLAB files are not needed for this step. Completed model outputs, valid negative results, and controls are included.

## Rebuild inputs and baseline predictions

Choose a fresh output directory, leaving the original evidence unchanged:

```sh
NEURAL_PHASE2_OUTPUT_DIR='/absolute/fresh/output' python /absolute/path/neural_decoding_phase2/code/prepare.py
```

The script downloads three public MATLAB files into the current project's `local/sources` if absent, verifies their SHA-256 values, recreates the participant/block splits and all prespecified calibration subsets, builds numerical inputs, and fits the comparison baselines. It also generates some baseline-only comparison cases; the actual Sol conditions are explicitly listed in `planned_calls.json`.

## Rerun Sol

After input preparation, use the same fresh output directory:

```sh
NEURAL_PHASE2_OUTPUT_DIR='/absolute/fresh/output' python /absolute/path/neural_decoding_phase2/code/run_planned.py
```

This makes real model calls through the existing Codex login. Set `CODEX_APP_RUNTIME_CLI_PATH` to the working binary if `codex` is not available on PATH. The runner requests the exact model slug and high effort, creates fresh ephemeral contexts, and does not substitute models. Completed conditions are skipped. New predictions are stochastic; there is no model sampling seed in this invocation. The supplied artifact records the actual successful predictions, rather than promising future byte-identical generations.

For new results, point `analyze.py` and `figures.py` at that output directory using the same environment variable. Do not call new predictions on these already-inspected recordings “untouched confirmation.”

## Statistical and scientific scope

- T12 original recordings are development data. Exact repetitions are new calls on the same 98 recordings. Calibration curves use three Sol draws at 2/4/7 examples per word; 14 uses one full calibration set and independent calls, including Phase 1. Conventional curves include twenty draws at smaller sizes.
- T16 was reserved before Phase 2 scoring. Its first block supplies calibration, its second supplies 49 evaluation word trials. The frozen recipe and hashes precede the first scoring. It tests the procedure with participant-specific calibration, not zero-calibration transfer between people.
- One new T12 session is exploratory after baseline inspection. Cross-session transfer uses old-session calibration with no new-session labels; within-new-session baseline comparisons are a different question.
- The same recordings are reused across methods, calibration draws, and model calls. Trial intervals and paired McNemar values are descriptive. Shared within-block label permutations preserve dependence across calls, conditional on within-block label exchangeability. Two participants cannot support a population-level accuracy estimate.
- Standardization and PCA fit calibration data only. Block centering uses unlabeled continuous activity within each block and is explicitly transductive. Query-mean alignment uses only the exact query vectors available to Sol; its hyperparameters are selected in calibration CV with the same operation. Neither is a streaming evaluation.
- Baselines use kNN, regularized shared-covariance LDA and Gaussian naive Bayes. LDA and GNB conservative tie-policy results are sensitivity analyses retained alongside the original first-grid policy. The identity-covariance limit is explicit when calibration folds contain one example per class.
- The rotation control preserves all 128 coordinates and freezes the original kNN/LDA parameters; it does not restandardize rotated dimensions. The selected conventional predictions are unchanged by the rounded rotation.
- All Sol prediction prompts prohibit tools and evaluator-file access. Event records provide the audit. This is an interface audit, not operating-system isolation under a separate user identity.
- `codex_version.txt` records the local CLI. The fixed slug identifies the deployment; a backend weight snapshot is not exposed. Pretraining exposure cannot be audited from this experiment.

`protocol.md`, `confirmation_freeze.json`, `research_log.md`, `literature.md`, source hashes, exact input files, predictions, and comparison configurations document the experimental choices. No Phase 3 decoding experiment or polished demo is included.
