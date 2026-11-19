# Phase 2 research log

## Initial hypotheses

The Phase 1 result could reflect reproducible in-context pattern matching, advantageous implicit handling of drift, a favorable model call, or access to whole-query-batch structure. It does not isolate speech production from cue memory. The immediate comparison is exact-call replication plus matched calibration curves; conventional baselines and temporal contrasts will guide expensive Sol calls.

## Intake and implementation decisions

T16 has two blocks, not the four found in T12. Before inspecting any T16 evaluation performance, revise its reserve to first block calibration (up to seven examples per word), second block untouched evaluation (49 word trials). The anatomical i6v array is called ventral6v in this file. For single-block calibration, use calibration-only stratified CV; no test labels enter selection. This is participant-specific adaptation in a second person, not zero-calibration cross-person transfer.

The two-example regime leaves one sample per class in an inner calibration fold. Use an explicit shared-covariance shrinkage LDA with a defined identity-covariance limit at zero within-class variance. Invalid kNN neighborhood sizes are skipped. The protocol states the formula and tie rule.

## First conventional results on development data

Across 20 calibration draws, raw-feature kNN accuracy at 2/4/7 examples per word is 25.5/30.9/34.1%; regularized LDA gives 30.3/37.2/39.5%. At fourteen examples per word (one full calibration set), kNN scores 30/98 and LDA 40/98. The original Sol result was 52/98, but new exact-call repetitions are pending. This suggests a potential shift-robustness advantage worth testing, not established superiority to conventional decoding generally.

Add Gaussian naive Bayes as one justified third comparator: Phase 1 Sol described variance-weighted pattern matching, and the source paper used Gaussian naive Bayes. This tests whether diagonal variance handling explains any apparent advantage over kNN/LDA. Tune smoothing only on calibration folds. Keep all initial and expanded comparator results transparent.

## Exact-call replication

The two fresh calls on the byte-identical Phase 1 raw-feature prompt scored 62/98 and 51/98, compared with the original 52/98. These are new predictions on previously evaluated recordings, not new held-out data. The phenomenon repeats, but call variability matters. At full calibration, the explicit fixed-shrinkage LDA's predictions match sklearn's implementation; baseline and Sol input arrays match exactly after rounding.

## Drift and temporal development probes

At fourteen examples per word, LDA improves from 40.8% raw to 67.3% with label-free within-block centering. GNB similarly improves from 39.8% to 66.3%; kNN improves from 30.6% to 50.0%. Delay-period LDA/GNB scores are 35.7/38.8%; same-trial post-minus-delay gives 36.7/40.8%. The s6v array is near chance (15-16%). These results favor substantial neural-word content plus nonstationary offsets; they do not identify production as the unique source. Send corrected, delay, and difference representations to Sol, without giving it baseline results.

## Confirmation freeze

The T16 confirmation uses two or seven labeled examples per word, because intake showed only two source blocks. First block provides participant-specific calibration; the second is a 49-trial untouched confirmation set. The raw window is unchanged. Include two raw seven-example calls, two two-example calibration draws, fresh missing-signal and within-block shuffled controls, and a block-centered secondary condition chosen from T12 development evidence. Freeze saved in confirmation_freeze.json before T16 scoring.

## Additional session, exploratory conventional evidence

The interleaved T12 session from April 2024 has ten blocks. First five calibrate; last five evaluate each behavior (70 word recordings per behavior). Baseline analysis makes these new-session recordings exploratory development data for later Sol calls. Imagined-to-imagined LDA gives 68.6% raw and 67.1% centered; imagined-to-attempted transfer gives 60.0% raw and 72.9% centered. Listening transfer is weak under the chosen window; this is not a definitive sensory-versus-speech dissociation because event timing/behavior differ. Transferring the old August 2023 calibration without any new labels gives 14.3% raw and 27.1% centered for kNN/LDA. Prepare one centered Sol transfer probe to test inference-time shift handling in a much harder setting. No claim of untouched confirmation for this adaptive probe.

## First untouched-participant result

After all T16 inputs and conditions were fixed, the first completed raw seven-example call (presentation draw 1) scored 34/49 (69.4%) on the untouched evaluation block. Raw LDA scored 22/49 (44.9%); centered LDA scored 38/49 (77.6%). Remaining frozen calls and controls continue without modification. This is within-session generalization after calibration in a second participant, not a classifier trained on T12 and transferred without new labels.

The first T12 two-example Sol draw scored 40/98 (40.8%). Other prespecified draws are pending; do not infer a calibration-efficiency advantage from a single favorable draw.

## Information-preserving representation test

Add a full orthogonal rotation of the 128 standardized development features. Unlike PCA12, it discards no dimensions. Rounded rotated inputs preserve the exact selected kNN/LDA predictions in this dataset. If Sol degrades while those baselines remain unchanged, that would favor coordinate readability/sparse patterns as a limitation, rather than information loss alone. This is one exploratory Sol call, not a new confirmatory endpoint. The rotation is fixed, random, and independent of labels.

## Baseline tie-policy sensitivity

The smallest calibration folds make all variance/shrinkage strengths observationally equivalent in calibration CV. A first-grid tie chooses weak shrinkage, which can make the conventional comparator unnecessarily brittle. Retain the initial baseline results and add a sensitivity policy that chooses the strongest regularization among exactly tied calibration-CV scores. This policy uses calibration scores only, with no evaluation-score selection. It is an adaptive diagnostic added after some initial test results, not a new preregistered confirmation result. Report both policies and require any low-data advantage to survive this stronger comparison. The same issue can affect Gaussian naive Bayes variance smoothing.

## Untouched T16 raw confirmations and controls completed

The two seven-example calls scored 28/49 and 34/49. The two two-example calibration draws scored 28/49 and 31/49. No-signal scored 7/49; within-block shuffled scored 6/49. All frozen raw predictions are now available, but the secondary centered call is pending. The low-data result survives comparison with the calibration-only conservative tie sensitivity in these observed draws. Do not treat the four overlapping evaluation calls as four independent participant cohorts or infer performance beyond this single new test block.

## Same-input unlabeled-batch alignment

Give the conventional methods access to the same unlabeled query batch as Sol, without using extra neural recordings: subtract the query-vector mean and add the calibration-vector mean. Apply the same label-free operation in calibration CV before choosing hyperparameters. This is explicitly transductive and assumes the cohort's word mixture is comparable; it is not a streaming result. It was motivated by the shift finding and added adaptively, not selected on T16 labels.

At full calibration, T12 aligned LDA/GNB reach about 63-64%, compared with Sol's 56.1% raw mean. At seven examples per word, T16 aligned LDA reaches 76.5%, compared with Sol's 63.3% raw mean. This removes an apparent general superiority over classical decoding. At two examples per word, T12 aligned conservative LDA reaches 40.5% versus Sol 37.8% on matched three draws; T16 Sol remains higher (60.2% versus aligned LDA 46.9% across two draws). Treat the residual T16 sparse-calibration advantage as a candidate effect requiring stronger replication, not a general conclusion.

## Final prioritization

Mean-aligned classical baselines explain much of the raw-feature advantage. Prioritize an independent repetition of the strongest same-representation result (centered T12 Sol 77/98 versus LDA 66/98). Complete calls already in progress and the resulting batch partition; defer additional alias, orthogonal-rotation, and no-recalibration cross-session Sol probes. No Sol result or capability conclusion is claimed for those deferred directions. All completed scientific outcomes remain included.
