# Neural decoding with GPT-6.1 Sol

Research on learning to interpret neural recordings during inference, without fine-tuning Sol or supplying a pretrained neural-to-language decoder.

[Read the technical blog](publication/essay.md) · [Reproduction instructions](research/phase3-sentence-reconstruction/REPRODUCE.md) · [V2 Remotion project](publication/film_v2/README.md)

The investigation started with seven-word classification. Sol's predictions depended on the neural recordings, but conventional classifiers remained competitive. Later experiments tested new word orders and sentence reconstruction, including symbolic inputs and inference-time numerical tools.

On 32 reserved T16 sentence recordings, tool-assisted Sol reconstructed 18 sentences exactly at 16.8% word error rate; an independent call reached 21.0%. The tested numerical decoder reached 35.7%. These results describe cued attempted speech within one recording session, with participant-specific calibration. They do not establish real-time decoding, unrestricted vocabulary or generalization to new sessions.

## Project layout

```text
research/
  phase1-word-decoding/          Initial discovery, controls and evidence
  phase2-decoder-comparisons/    Replication, calibration and baselines
  phase3-sentence-reconstruction/
                                Sentence methods, code and experimental evidence
publication/
  essay.md                      Technical blog source
  essay.html                    Self-contained reading copy
  fonts/                        Bundled fonts and licenses
  film_v2/                      Editable Remotion source, assets and QA
  VIDEO_VERSIONS.md              Notes on local video versions
local/                          Original recordings and scratch space; Git-ignored
```

Completed negative scientific results are retained. Failed infrastructure attempts, unused downloads, duplicate scratch copies and superseded drafts were removed. The V1 Remotion project was removed; V2 is retained. Rendered videos remain local and are not uploaded to GitHub.

## Reproduce

Run commands from the repository root. Exact research package versions are in `requirements.txt`.

```sh
uv venv .venv --python 3.15.2
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tools/fetch_sources.py --dataset phase3
.venv/bin/python research/phase3-sentence-reconstruction/code/build_essay.py
```

The reproduction guide explains the saved analyses, baseline replay and independent model calls. All original model experiments used authenticated Codex with `gpt-6.1-sol` at high reasoning effort. Analysis and blog-build commands do not invoke models.

Historical prompts, cases and invocation records retain the original machine paths as provenance. Portable analysis code resolves those paths to archived traces. Prediction/configuration locks and snapshots of the frozen baseline code are retained. Local recordings, installed packages, rendered videos and large reproducible computation caches are excluded from Git.

Original data are credited in each study's source documentation. Fonts and third-party source references retain their license notices; no ownership of the source recordings is claimed.
