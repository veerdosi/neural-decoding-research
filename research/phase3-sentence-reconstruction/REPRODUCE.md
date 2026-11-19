# Reproducing the research

Run commands from the project root, `/Users/veerdosi/Documents/code/github/neural-decoding-research`, unless a different working directory is shown. The study directory is `research/phase3-sentence-reconstruction/`. Blog and video assets are in `publication/`.

## What is preserved

- `cases/`, `prompts/`, `responses/`: frozen splits, inputs and authenticated Codex invocation/event records. Only completed invocations are counted as results. A completed negative scientific result is retained.
- `evaluation/`: per-recording predictions, targets, paired analyses, audits and conventional baselines. The original configuration/prediction locks must remain unchanged.
- `discovery_probes/`: image, symbol, unseen-word, fusion and text-correction inputs, code and outputs. Several of these are explicitly comparison stages rather than independent direct-decoding methods.
- `agentic_traces/`: archived completed working directories, including model-written numerical code and calibration-derived references. `evaluation/agentic_trace_archive.json` maps original working directories to archives and hashes their files. The first exploratory call also has `agentic_development_trace/`.
- `figures/`: PNG/SVG figures and source-data/provenance JSON.
- `source_registry.json`, `source_readmes/`: source locations, hashes and acquisition/version notes. Original MAT recordings are in `local/sources/`, outside the deliverable directory.
- `publication/`: blog, shared fonts and the V2 Remotion project. Rendered videos remain local and are excluded from Git. The superseded V1 Remotion project was removed.

## Environment

The current workspace uses Python3.15.2 on macOS arm64, NumPy/SciPy/scikit-learn and Clang. Exact Python package versions are in `requirements-research.txt`.

```sh
uv venv .venv --python 3.15.2
uv pip install --python .venv/bin/python -r research/phase3-sentence-reconstruction/requirements-research.txt
```

Skip environment creation in the existing workspace. Numerical routines should use one BLAS thread. The fast DTW helper is C source in `code/sentence_dtw.c`; its current compilation command targets a macOS dynamic library. Another operating system needs the corresponding shared-library compiler flags.

## Recompute saved-result analyses

These commands make **no model calls**. They rescore completed predictions against the saved source recordings and regenerate analyses/figures. They do not create new predictions or untouched evaluation data.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/audit.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_t16_confirmation.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_t16_tool_contribution.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_t16_replication.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/audit_t16_confirmation.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_neural_alphabet_confirmation.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_complementarity.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/analyze_discovery_probes.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/research_figures.py
```

`audit.py` covers the original sequence experiment only. The T16 method audit covers observable commands, source reconstruction, calibration-only PCA and calibration interval provenance. Neither audit certifies population generalization or undisclosed model pretraining.

## Recompute a conventional prediction without changing the original lock

The full-feature workflow has two configurations: the T12-selected generic transfer and the T16 calibration-selected variant. Both use the same80 calibration and32 evaluation recordings as Sol. The latter configuration is in `evaluation/algorithm_sentence_baselines/full_evidence/generic_sentences_t16_fresh_dev_configuration_lock.json`.

Use a new tag to preserve the original blinded prediction files:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/sentence_full_evidence_comparison.py predict_generic --case sentences_t16_confirm --tag replay_generic
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/sentence_full_evidence_comparison.py evaluate_generic --case sentences_t16_confirm --tag replay_generic
```

For the T16-calibrated configuration, add `--configuration generic_sentences_t16_fresh_dev_configuration_lock.json` and use another new tag. This recomputes deterministic predictions. Calibration CV is available in the same script; do not overwrite the historic locks when experimenting.

The categorical HMM comparator is `code/neural_alphabet_baseline.py`. Its word-state emissions use only the same discrete codes supplied to Sol. Hyperparameter selection uses calibration blocks; no test targets enter fitting. The calibration-only representation is shared across CV folds, as documented in its output.

## Replay the numerical portion of a Sol-authored procedure

Copy the relevant archived case to a new scratch directory before executing its scripts. Saved archives are evidence and should remain unchanged. For the first T16 confirmation, `exemplar_decode.py`, `balanced_dtw.py` and `dtw_exemplars.py` preserve the three numerical sequence stages.

These scripts depend on Sol's saved calibration timing estimates. Replaying them reproduces the numerical procedure **conditional on those estimates**. It does not independently recreate Sol's choices or final interpretation. The latter requires a fresh model inference and is not deterministic.

## Make a genuinely independent model call

All research model calls use **`gpt-6.1-sol`, high reasoning effort, through authenticated Codex**. No direct model API route or API key is implemented. `CODEX_APP_RUNTIME_CLI_PATH` must point to the authenticated installed Codex executable; the desktop environment supplies it here.

Use a new case name and clean working directory. For example:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/agentic_cases.py prepare sentences_t16_reproduction --source t16.2024.04.29_attempted_50words.mat --cal-blocks 1 4 --query-block 6 --start 8 --n 32 --roi
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/agentic_cases.py run sentences_t16_reproduction
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/agentic_cases.py evaluate sentences_t16_reproduction
```

This gives new predictions on already evaluated recordings, not new confirmation data. Generic numerical helpers are supplied, but the first call's word references and predictions are not.

For the symbolic method, prepare an independent case using the same raw source case:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/phase3-sentence-reconstruction/code/discovery_probes.py prepare_symbols neural_alphabet_reproduction --source-case sentences_t16_reproduction
.venv/bin/python research/phase3-sentence-reconstruction/code/discovery_probes.py run neural_alphabet_reproduction
.venv/bin/python research/phase3-sentence-reconstruction/code/discovery_probes.py evaluate neural_alphabet_reproduction
```

The fixed symbolic representation uses smoothing, PCA32, per-component scaling,64 calibration-only KMeans clusters and consistently permuted IDs. No word labels train those clusters. The Sol prompt prohibits tools; event logs verify whether that condition held.

The visual and unknown-word probes are preserved in `discovery_probes/`. Do not rerun the general `prepare` action over their completed evidence directories. Create fresh case directories when changing a method. The unknown-word source has no whole-word calibration/evaluation overlap; it uses a generic pronunciation dictionary and never uses its audio fields.

## Interpretation rules

- Previously inspected recordings are development data for new methods. The later fusion and text-only attribution comparisons therefore carry development labels.
- Independent calls do not increase the number of independent recordings. Controls and repeats share the underlying cohort.
- A target-informed oracle is a complementarity bound, never an achieved decoder.
- T16cal cached assembly time is not an independent end-to-end runtime. Common preprocessing and calibration CV have different timing scopes from a model inference.
- “No Sol fine-tuning” does not mean no task-specific procedure was fitted. Sol constructed calibration-derived numerical references during inference.
- Source mean correction uses whole unlabelled blocks; these are offline, transductive experiments.

The essay is regenerated from `publication/essay.md` by this study's `code/build_essay.py`. Its HTML embeds fonts and figures for offline reading. Relative evidence links require the accompanying study directory.

## Relocated evidence

Historical JSON, prompts and model event records retain their original paths and hashes. `code/paths.py` resolves completed sandboxes to the archived `agentic_traces/` directories. The initial exploratory sandbox resolves to `agentic_development_trace/`. Do not edit frozen evidence to repair old absolute paths. `evidence_manifest.json` records the original file hashes. Large deterministic retrieval caches and intermediate dictionary score arrays remain local and are ignored by Git; saved prediction locks, final results and code are retained.
