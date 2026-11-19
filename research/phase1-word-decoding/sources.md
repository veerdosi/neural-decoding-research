# Sources inspected

1. [Kunz et al., Cell (2025), full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360486/). Figure 1 and Methods: single-word task, disappearance of text at go, original supervised decoding results. This pilot does not reuse the published decoder.
2. [Original dataset on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j). Version 2, 30 July 2025, task bundle descriptions and anonymized participant codes.
3. [Hugging Face mirror supplied by the user](https://huggingface.co/datasets/raei/Kunz-2025-InnerSpeechMotorCortexImplicationsSpeechNeuroprostheses). An unofficial mirror of the Dryad files. Only one task/participant is used for prediction in this pilot.
4. [Original task-specific README](https://huggingface.co/datasets/raei/Kunz-2025-InnerSpeechMotorCortexImplicationsSpeechNeuroprostheses/blob/main/isolatedVerbalBehaviors/isolatedVerbalBehaviors_readme.txt). Neural feature definitions, MATLAB indexing, block/epoch fields and word set. A local reference copy is included in `sources/`.
5. [Original analysis code](https://github.com/nptl-stanford/inner_speech/blob/66b7948f3134971eba51431e8829e475abf362dd/analysis/Fig1EF_7Behavior7WordDecoding.m). Inspected at commit `66b7948f3134971eba51431e8829e475abf362dd`. Confirms the i6v channel ordering and original feature concatenation. The pilot uses its own calibration-only normalization and fixed time window; it does not execute the original classifier.

Model availability evidence is local and empirical: `model_catalog_entry.json`, `codex_version.txt`, `responses/access_check.jsonl`, and every condition's explicit Codex invocation. No public-API model availability is inferred from Codex availability.
