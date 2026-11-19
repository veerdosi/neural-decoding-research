Analysis was performed only on inputs.json and neural_inputs.npz plus the numerical helpers originally present in this directory. No external data, network access, pretrained decoder, agent delegation, or parameter fitting of a neural-to-word model was used.

Numerical evidence

The PCA is estimated from calibration frames only. Features are projected into that basis, components 0 and 1 are removed, trajectories are smoothed with Gaussian sigma 1.2 bins, and frames are normalized. The supplied interval search interpolates trajectories to eight samples and ranks cosine similarity. Subsequence DTW allows (query, calibration) increments (1,1), (2,1), and (1,2), with the supplied 0.035 slope penalty.

The main raw evidence is local_drop2.json, local_dtw_drop2.json, edge_drop2.json, whole_phrase_evidence.json, and targeted_evidence.json. Corresponding text reports summarize these files. fast_dtw.c implements the supplied recurrence in C; its subsequence scores, bounds, and path were checked against the original Python helper on a deterministic numerical example. Compilation uses cc -O3 -shared -fPIC fast_dtw.c -o fast_dtw.so.

Calibration-only interval retrieval validation compared the original representation with the provided 64- and 32-component alternatives and whitening. It also checked per-record centering and a temporal high-pass variant. The original drop2 representation performed best on the approximate interval ledger and was retained. These figures are descriptive checks against manually estimated boundaries, not estimates of query reconstruction accuracy.

Interpretation and numerical assembly

build_ledger.py contains approximate, manually inferred calibration boundaries. They were interpreted from repeated and overlapping transcript fragments. Some boundaries remain rough, especially in recordings with long initial pauses. These annotations are analysis choices, not supplied ground truth.

exemplar_assembly.py and dtw_assembly.py perform nearest-exemplar, semi-Markov numerical assembly without a grammar or language model. They use a minimum interval length of five bins and may merge short words or assign spurious trailing words. Their unedited token sequences are saved separately in algorithm_only_predictions.json and algorithm_only_dtw_predictions.json, with their interval paths and lattice arrays. Neither file is the final answer.

save_reconstruction.py contains the assistant's final interpreted token sequences and the reasoning for changes from numerical assembly. It writes final_reconstruction.json and reconstruction_evidence.json. The latter combines approximate query timing notes with unrestricted retrieved matches. Inferences such as brief inserted words, unresolved initial intervals, and decisions about repetitions are explicitly described there. Word-level confidence varies considerably. Numerical similarity is evidence of trajectory resemblance; it does not establish lexical identity by itself.

Reproduction

Use the Python executable supplied by the user and OPENBLAS_NUM_THREADS=1. In order, run prepare_pca.py; local_retrieval.py drop2; edge_retrieval.py drop2; fast_helpers.py; build_ledger.py; exemplar_assembly.py; dtw_assembly.py; phrase_evidence.py; targeted_evidence.py; save_reconstruction.py. The human interpretation is preserved literally in the last script so it can be inspected rather than presented as algorithm output. The original calibration inspection command is recorded in the generated inspect_drop2_c026_c073_c067_c001_c062_c058_c039_c071_c052_c050.json filename and uses inspect_intervals.py with those IDs.
