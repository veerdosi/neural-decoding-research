# GPT-6.1 Sol neural decoding pilot

All model calls use the authenticated Codex CLI with `-m gpt-6.1-sol`. No direct API requests, fine-tuning, pretrained neural encoders, or externally fitted supervised decoders are used.

## Contents

- `code/pilot.py`: download, deterministic feature construction, prompts, Codex invocation, scoring.
- `code/similarity.py`: label-free distances to individual calibration recordings; no external voting or word decisions.
- `config.json`: dataset hash, split, features, seed, PCA variance, duplicate audit.
- `prompts/`: exact model inputs. Evaluation labels never appear in them.
- `responses/`: final model outputs, invocation arguments, event logs, and errors.
- `evaluation/ground_truth.json`: evaluator-only trial mapping and labels.
- `evaluation/prepared.npz`: standardized features, PCA transform, split and source-trial mappings.
- `evaluation/metrics.json`: accuracy, intervals, controls, paired comparisons and confusion matrices.
- `research_log.md`: hypotheses, choices, results and qualifications.
- `sources/`: source documentation and a reference copy of the original analysis script.

## Reproduce

Use a Python environment with the pinned packages in `requirements.txt`. The original run used Python 3.15.2. A recent Python with compatible NumPy and SciPy can run the code. Authenticate Codex normally; do not place credentials in this directory. Set `CODEX_APP_RUNTIME_CLI_PATH` to the working Codex binary if `codex` is not on PATH.

From the parent project directory:

```sh
python research/phase1-word-decoding/code/pilot.py prepare
python research/phase1-word-decoding/code/pilot.py run --condition pca12_icl
python research/phase1-word-decoding/code/pilot.py run --condition raw128_icl
python research/phase1-word-decoding/code/pilot.py run --condition pca12_shuffled
python research/phase1-word-decoding/code/pilot.py run --condition raw128_shuffled
python research/phase1-word-decoding/code/pilot.py run --condition raw128_zero_shot
python research/phase1-word-decoding/code/pilot.py run --condition no_neural
python research/phase1-word-decoding/code/similarity.py
python research/phase1-word-decoding/code/pilot.py run --condition similarity_icl
python research/phase1-word-decoding/code/pilot.py run --condition similarity_shuffled
python research/phase1-word-decoding/code/pilot.py evaluate
```

Each `run` starts a fresh ephemeral Codex context at high reasoning effort. The inference prompt prohibits tools and file access. Scoring rejects model-issued tool calls. Responses are stochastic; no sampling seed is available through this invocation. Repeating `run` overwrites that condition's stored output, so copy the bundle first to retain the original evidence. Preparing again deterministically recreates the same split and prompts and downloads only the selected 32 MB MATLAB file if absent.

The working directory used for calls is `work/model_sandbox`, which contains no evaluation files. File exclusion is audited through the tool-free event logs, rather than enforced with a separate operating-system identity. The default Codex system context remains present and is included in token accounting. No test feedback is returned to prediction calls.

Saved metrics can be recomputed without downloading the dataset or calling Codex, provided NumPy and SciPy are installed. Trial-level intervals are descriptive; two test blocks cannot establish generalization across participants or days.
