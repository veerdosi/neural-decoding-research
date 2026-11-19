## Mission

Continue the existing research project using GPT-6.1 Sol.

Phase 1 established that Sol can extract word-related information from neural recordings. Phase 2 replicated this across participants but found no consistent advantage over conventional classifiers.

**Go beyond classification and demonstrate that Sol can reconstruct meaningful language from neural activity, including ordered word sequences or combinations not explicitly seen during calibration.**

Act as an ambitious, autonomous research scientist. Discover a genuinely interesting capability and present the strongest supported result in a polished video with length determined by the discovery story, suitable for X. Aim for an impressive result in the spirit of the Bitter Lesson: a general-purpose model acquiring useful structure from data and inference-time computation, with minimal task-specific engineering.

Read the existing reports, code, and experimental results before proceeding. Reuse the infrastructure.

## 1. Push the capability

Start with the Kunz 2025 intracortical dataset. Investigate other public neural speech datasets if they offer a substantially better opportunity.

Progress from isolated words to ordered word sequences, and to sentences where feasible. Prioritize held-out sequences or combinations that were not explicitly supplied as calibration examples.

Explore signal representations, temporal reasoning, compositional inference, computational tools, or other justified approaches. Do not limit the investigation to conventional prompting or a predetermined method.

**Sol must decode neural information without fine-tuning or relying on a separately trained neural-to-language decoder.** Deterministic signal processing, labeled in-context examples, and inference-time computation are allowed.

Use `gpt-6.1-sol` through the existing authenticated Codex setup with high reasoning effort. Do not silently substitute models.

## 2. Research and iterate

Form hypotheses, run real experiments, inspect results, and improve methods according to the evidence. If a direction plateaus, investigate why and try a substantially different approach instead of repeatedly tweaking prompts.

Establish that predictions depend on the correct neural recording, rather than language priors, cue leakage, memorized utterances, or evaluation artifacts. Use suitable no-signal, shuffled-recording, and held-out-combination controls. Compare with relevant conventional methods when assessing a claimed capability or advantage.

Prioritize meaningful progress toward compositional or open-vocabulary decoding. Small gains on the original seven-word classification task are not the objective.

Favor general learning and computation over hand-engineering the answer into the representation. Keep deterministic preprocessing auditable and distinguish it from the information Sol actually infers.

You have autonomy over research methods and experimental priorities. Stay focused on discovering a useful and scientifically distinctive capability. A breakthrough is an ambition; claims must follow controlled evidence.

## 3. Produce the video

**Deliver a polished, shareable video with no strict duration requirement showing the most impressive capability supported by the actual experiments.**

An interactive application or website is not required. Do not spend effort building one.

The video should make the result immediately understandable and show:

- Real recorded neural activity.
- Sol's decoded words or ordered sequences, or sentences if achieved.
- Ground truth revealed after the prediction.
- A focused, visually impressive showcase using actual verified outputs, with minimal explanatory text.

Put controls, uncertainty, conventional comparisons and scientific limitations in the accompanying essay/evidence. They remain required research, but need not appear in the showcase video. Do not invent results or make unsupported performance claims.

Use actual model outputs. Clearly distinguish replayed recordings and completed inference from live decoding. Do not imply that playback duration is the model's inference latency.

Make the video visually compelling and suitable for X. If full sentence reconstruction is unsuccessful, show the most ambitious capability that the evidence supports. Do not fabricate success or abandon the video.

## 4. Final deliverables

Complete the investigation and provide:

1. **Shareable video:** A polished discovery story and capability showcase, suitable for sharing; length follows the content.
2. **Research results:** What Sol achieved, including relevant conventional comparisons and neural-dependence controls.
3. **Reproducible code and evidence:** Experiments, configurations, prompts, predictions, and evaluation outputs.
4. **Concise findings:** What was discovered, why it matters, what remains unresolved, and the strongest defensible scientific claim.

Use subagents selectively for independent research, engineering, or evaluation tasks when they provide a concrete advantage. Retain ownership of the research decisions and scientific conclusions.

Do not stop after planning, feasibility analysis, or a benchmark report. Execute the research and finish the video and supporting evidence.

## Guiding principle

**Uncover an unexpected capability of frontier intelligence, rather than merely reproducing what conventional classifiers already do.**

Be creative, skeptical, persistent, and ambitious. Push toward genuine brain-to-language reconstruction while maintaining rigorous controls.


## Latest user direction: research ambition and film craft

Do not treat the current two-word sequence result or one failed sentence prompt as exhausting the research mission. Push the most ambitious feasible neural-to-language capability through substantive alternative methods, inference-time computation, temporal alignment, calibration data and suitable public speech recordings. Follow evidence rather than promising a breakthrough. Keep the film provisional while the substantive research is still changing.

Art direction: Scientific Instrument Cinema — authentic mission-control telemetry, experimental neuroscience and Swiss editorial precision. Use the official remotion-dev/skills best practices and Remotion for the final film. IBM Plex Sans and IBM Plex Mono; near-black, warm white, muted grey and one restrained signal-orange accent. Actual neural data is the spectacle. Precise restrained motion, generous negative space and subtle intentional sound design that is optional to comprehension. No generic cards, dashboards, gradients, glows or illustrative fake waves.

The video should showcase the ability and memorable real prediction/truth reveals, with very little text. Controls and limitations belong in the essay, not explanatory video sections. Keep a small recorded-data/inference-replay identifier and do not imply live inference. Render, inspect representative frames and revise visually weak work. Delete obsolete renderer code, previews and superseded exports while retaining scientific evidence and reused authentic assets.

Video versioning: preserve the existing Remotion showcase and its artifacts as v1 (`video_v1.mp4`, `film_v1/`). The fuller successor is v2 (`video_v2.mp4`, `film_v2/`); do not overwrite v1 or present it as the completed fuller investigation.


## Phase 3 contribution clarification

Sol may use general-purpose numerical tools, including PCA, smoothing, similarity search, DTW and other computational methods. Avoid separately trained or pre-existing specialized neural-to-language decoders doing the decoding for it. Do not unnecessarily restrict its computational freedom or abandon the successful approach.

For the existing approximately26%WER development result, compare the complete Sol-assisted approach against the same retrieval/alignment pipeline without Sol. Distinguish the information and reconstruction already determined by algorithms from Sol's contribution to interpretation, composition or improved decoding. Use matched pipeline evidence, appropriate non-LLM assembly baselines, calibration-only tuning and untouched evaluation. These comparisons complement other conventional baselines and should inform further ambitious decoding research.

Only make v2 after the research investigation is complete. The video should combine a compelling scientific discovery story with an impressive showcase of what GPT-6.1 Sol actually does with neural signals, naturally conveying what was found, why it is surprising and why it matters. There is no strict duration requirement. Preservev1. Do not rush video production ahead of research.


## Discovery-first reassessment

Do not remain in a retrieval/DTW local optimum. Regardless of T16 confirmation outcomes, experimentally probe a few fundamentally different mechanisms that exploit Sol's general-purpose intelligence. Balance rigorous attribution with ambitious capability discovery. One meaningful contribution comparison is sufficient for each promising approach; reserve extensive validation for the strongest discoveries. Do not spend disproportionate effort validating marginal gains. Let evidence choose the next direction. The objective remains a scientifically distinctive neural decoding ability that conventional approaches struggle to achieve.

## Latest user sequencing: essay review before v2

Finish the substantive research comparisons and share the findings first as a technical blog-style essay, with figures, methods, contribution attribution and evidence. The user will read it and give instructions for the video's story, structure and inclusions. Await that feedback before starting v2 video production. Preserve v1. This review gate changes the sequence of work, not the research objective or the eventual video deliverable. Do not infer approval to start v2 from elapsed time or an automatic goal continuation.
