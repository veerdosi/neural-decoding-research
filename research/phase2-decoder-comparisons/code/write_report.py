import json
from pathlib import Path
import numpy as np
from experiment import ROOT
E=ROOT/'evaluation'
s=json.loads((E/'phase2_summary.json').read_text())
def load(name,suffix='sol'):
 return json.loads((E/f'{name}.{suffix}.json').read_text())
def pct(x):return f'{100*x:.1f}%'
rows=[]
for n in [2,4,7,14]:
 r=s['development_calibration_curve'][str(n)]
 rows.append(f"| T12, {n} | {pct(r['sol_mean'])} | {pct(r['knn']['matched_mean'])} | {pct(r['lda']['matched_mean'])} | {pct(r['mean_aligned_lda']['matched_mean'])} |")
for n in [2,7]:
 names=[f'confirm_raw_n{n}_draw{i}' for i in [0,1]]
 vals=[np.mean([load(c)['accuracy'] for c in names])]
 vals += [np.mean([load(c,'baseline')[method]['accuracy'] for c in names]) for method in ['knn','lda']]
 vals += [np.mean([load(c,'baseline_mean_aligned')['lda']['accuracy'] for c in names])]
 rows.append('| T16, '+str(n)+' | '+' | '.join(pct(v) for v in vals)+' |')
table='| Participant, examples/word | Sol | kNN | Regularized LDA | LDA + query-mean alignment |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)
n_calls=len(s['sol_calls']);n_predictions=sum(x['n'] for x in s['sol_calls'].values());chunk=s['chunk_protocol']
text=f'''# Phase 2 research report

18 November 2026. All Sol calls used `gpt-6.1-sol` through authenticated Codex at high reasoning effort.

**The neural-dependent in-context classification effect replicated. A broad decoding advantage over conventional methods did not emerge.** Sol was strong against static classifiers on shifted raw features, but simple label-free batch alignment closed or reversed much of that gap. Performance also depended on presentation and inference allocation. Speech-specific decoding and a fundamentally stronger BCI approach remain unestablished.

This report contains {n_calls} completed Phase 2 model calls and {n_predictions:,} predictions. They reuse **147 unique evaluation recordings**: 98 T12 development trials and 49 reserved T16 trials. Repetitions are not additional independent recordings. The queue is cleared; no remaining tests are running.

## A. Reproducibility

The exact Phase 1 full-feature prompt scored **52/98, 62/98 and 51/98** across the original call and two fresh calls: mean 56.1%, range 52.0–63.3%. This is replication on previously evaluated recordings.

The reserved T16 protocol was frozen before first scoring: block 9 supplies participant-specific calibration; later block 17 supplies 49 word trials. With seven examples per word, two independent calls/presentation orders scored **28/49 and 34/49**. Two examples per word gave **28/49 and 31/49** on the same test recordings. The fresh no-signal control scored **7/49**; shuffled recordings scored **6/49**, versus 14.3% chance. In the shuffled call, 31/49 predictions matched the actual donor recording's word. This supports recording-linked information rather than query-ID or language-prior guessing.

This is replication of a learning procedure in a second participant **with that participant's calibration**. It is not zero-calibration transfer between people. A third recording session was analyzed with conventional methods only; there is no completed Sol cross-session transfer result.

Shared within-block label permutations preserved dependence across repeated calls. Neither T12 nor T16's aggregate correct-recording score was reached in 20,000 permutations (plus-one Monte Carlo p=0.000050 for each, conditional on within-block label exchangeability). No participant-population accuracy estimate is justified by two people. Per-call exact trial intervals, block outcomes and descriptive paired comparisons are saved separately.

## B. Distinctiveness and calibration efficiency

Every direct comparison used the same neural features, selected calibration examples and held-out recordings. Standardization used only the selected calibration inputs. kNN and shared-covariance regularized LDA were tuned on calibration data alone. Gaussian naive Bayes was added because the original study and Sol's stated comparison strategy made diagonal variance weighting a relevant alternative. Baselines used twenty draws at smaller T12 calibration sizes; the table uses the same three draws as Sol. Fourteen examples uses one complete calibration set and repeated Sol calls. T16 uses two draws/presentations.

{table}

These are observed means, not population confidence intervals. The primary table retains the original calibration-CV tie policy; a conservative regularization tie-policy sensitivity is also included. Both policies leave the main conclusion unchanged. The smallest folds contain one example per class, so covariance strength is underidentified; an explicit identity-covariance limit avoids invalid fitting.

The aligned baseline subtracts the unlabeled query-vector mean and adds the calibration-vector mean. It sees **only the same vectors available to Sol**, uses no query labels, and applies alignment inside calibration CV too. It is transductive and benefits from this experiment's comparable calibration/query word mixtures. It is not streaming decoding.

At full T12 calibration, aligned Gaussian naive Bayes reached **63/98 (64.3%)**, above Sol's full-batch mean. T16's residual two-example gain over aligned LDA was 4/49 and 9/49 in the two draws; descriptive two-sided paired p-values were 0.289 and 0.049, before multiplicity and session-correlation concerns. This is a candidate sparse-calibration effect, not established general superiority.

The completed smaller-batch protocol scored **{chunk['correct']}/98 ({pct(chunk['accuracy'])})** across seven 14-query calls, with exactly the original calibration/header and the same recordings. Thus a large unlabeled query cohort was not necessary for the observed effect. The change also increased total inference work and repeated calibration processing, so it does not isolate attention, reasoning budget or batch size as a causal mechanism.

## C. What information is extracted, and what limits it?

- **Word information precedes instructed execution.** Sol decoded the pre-go cue period at **52/98 (53.1%)**, close to the full-batch post-go mean. These recordings can encode reading, preparation, remembered cues or cue-triggered inner speech. Identifying the cued word does not uniquely identify speech production or independently verify subjective thought.
- **Drift is consequential.** Label-free within-block centering raised T12 Sol to **77/98 (78.6%)**, versus LDA's 66/98 (67.3%) on that representation. In T16, centered Sol scored **36/49 (73.5%)**, below LDA's 38/49 (77.6%). The T12 centered Sol gain has one completed call and is not independently replicated.
- **Information is distributed across features.** Top-12-PC LDA and remaining-PC LDA each scored 36/98. Useful word information is not confined to the highest-variance directions. This supports an information-loss explanation for part of PCA's weakness, but does not establish Sol's internal algorithm. Threshold crossings and spike power alone remained informative; the other T12 cortical array was near chance.
- **Session changes remain harder.** On the later T12 interleaved session, within-session imagined-word LDA reached 68.6%. Transferring the old session's calibration without new labels gave 14.3% raw and 27.1% after centering. These are conventional-model results; Sol transfer was not evaluated. Different session/task conditions prevent a clean single-factor causal explanation.

All prediction calls were tool-free. No fine-tuning, gradient updates, external trained neural decoder or pretrained neural encoder entered the Sol path. The frozen deployment's historical pretraining exposure cannot be audited. Model explanations about pattern matching are self-reports, not verified internal mechanisms.

No calibration/evaluation trial overlap, exact feature duplicates, or repeated block cue order was found. Query inputs contain numerical features and random IDs, with no evaluation labels or stimulus metadata. The whole vocabulary is shared between calibration and evaluation; this tests new recordings of known words, not unseen-word or sentence reconstruction.

## D. Research significance

The strongest contribution is a controlled demonstration that a frozen general-purpose model can acquire participant-specific neural-label associations from small labeled contexts, with reproducible recording-dependent predictions in two participants. The comparison also identifies substantial contributions from shift handling, query-batch information and presentation/inference allocation. It does not establish a novel speech-specific code or consistent superiority over trained decoding.

Inference-only neural interpretation is not a new concept: earlier work tested GPT in-context learning on numerical EEG features ([Sano et al., 2024](https://arxiv.org/html/2408.07822v1)), and retrieval-conditioned VLMs have classified EEG plots ([Li et al., 2026](https://arxiv.org/html/2601.17844v1)). Meta-trained fMRI in-context adaptation is another distinct precedent ([Nan et al., 2026](https://arxiv.org/abs/2604.08537)). The original [Kunz dataset and study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360486/) already establish supervised neural word information. Our evidence narrows the claim to this deployment, intracortical task, adaptation behavior and controlled failure boundaries.

The practical cost is high: the retained calls report approximately **2.15 million input tokens and 0.63 million output tokens, including reasoning**, with median operational latency about **8.5 minutes per call**. Classical tuning and prediction took milliseconds to small fractions of a second locally. These are different execution environments, not a hardware-normalized benchmark; completed-turn token usage and per-case timings are provided. No real-time BCI performance was demonstrated.

## E. Phase 3 recommendation

**Recommend a small compositional temporal-decoding test with matched cue controls.** Test whether Sol can use a few neural examples of component words to reconstruct held-out word sequences, including combinations absent from calibration. Compare with strong factorized temporal/template decoders, matched unlabeled-data access and an explicit inference budget. The existing Kunz speech, verbal-memory and visual-memory tasks provide a feasible starting point: direction words/sequences and matched arrow/movement tasks with contrasting rehearsal strategies are documented. First verify that the recordings actually contain position-specific information; no model can recover absent signal. This would test a structured generalization ability that seven arbitrary labels do not expose. If successful, an honest demonstration could show unseen sequence composition from recorded neural data with visible controls.

The alternative is a focused sparse-calibration study on untouched sessions/participants under controlled shifts, testing whether the T16 two-example residual survives stronger baselines. It is easier technically but has less breakthrough/demo potential. No Phase 3 experiment or polished demo has begun.

## Completed negative results and limits

Zero-shot Phase 1 was at chance. Static similarity/PCA representations underperformed the full features. The other cortical array and several listening-transfer comparisons were weak; listening timing differences prevent interpreting those as absence of a sensory representation. Classical old-to-new-session transfer was poor. No successful Sol result exists for arbitrary label remapping, full orthogonal rotation, or zero-new-label cross-session transfer; those further probes were not pursued. A repetition of the centered T12 Sol condition was also not completed, so its single-call advantage remains provisional.

See `research_log.md`, `protocol.md`, `confirmation_freeze.json`, `evaluation/phase2_summary.json`, `evaluation/successful_experiments.csv`, exact prompts, model responses, baseline configurations, and the README reproduction commands. All completed scientific results are retained.
'''
(ROOT/'report.md').write_text(text)
print(ROOT/'report.md')
