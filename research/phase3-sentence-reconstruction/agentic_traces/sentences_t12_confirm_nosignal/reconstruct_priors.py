"""Reproducible guesses when every query trajectory has been withheld.

Run: OPENBLAS_NUM_THREADS=1 /Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python reconstruct_priors.py

No neural features are fitted or scored. Novel transcript candidates are formed
by joining a calibration prefix to another calibration suffix at an identical
contiguous span. Whole-candidate support counts are a text-only empirical prior,
not neural word scores. The only query conditioning is recording length.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    meta = json.loads((root / "inputs.json").read_text())
    raw = np.load(root / "neural_inputs.npz")
    calibration = meta["calibration"]
    originals = {tuple(c["tokens"]) for c in calibration}
    vocabulary = sorted({w for t in originals for w in t})
    max_words = max(map(len, originals))
    min_words = min(map(len, originals))
    checks = []
    for q in meta["queries"]:
        x = raw[q["id"]]
        checks.append({"id": q["id"], "shape": list(x.shape),
                       "nonzero_values": int(np.count_nonzero(x)),
                       "all_finite": bool(np.isfinite(x).all())})
    if any(q["nonzero_values"] for q in checks):
        raise ValueError("This prior-only procedure assumes withheld zero queries.")

    # A source pair contributes once to a candidate even if several shared spans
    # generate that candidate. Do not add language-model or per-word scores.
    support = defaultdict(dict)
    for left in calibration:
        a = tuple(left["tokens"])
        for right in calibration:
            if left["id"] == right["id"]:
                continue
            b = tuple(right["tokens"])
            for i in range(1, len(a)):
                for j in range(len(b) - 1):
                    for k in range(1, min(len(a) - i, len(b) - j - 1) + 1):
                        if a[i:i+k] != b[j:j+k]:
                            break
                        candidate = a[:i+k] + b[j+k:]
                        if candidate in originals or not min_words <= len(candidate) <= max_words:
                            continue
                        pair = (left["id"], right["id"])
                        support[candidate].setdefault(pair, {
                            "left_id": left["id"], "right_id": right["id"],
                            "left_tokens": list(a), "right_tokens": list(b),
                            "left_prefix_end": i+k, "right_suffix_start": j+k,
                            "shared_tokens": list(a[i:i+k]),
                        })

    total_support = Counter()
    frame_lengths = defaultdict(list)
    for candidate, pairs in support.items():
        total_support[len(candidate)] += len(pairs)
    for c in calibration:
        frame_lengths[len(c["tokens"])].append(raw[c["id"]].shape[0])

    # Fixed 0.4-second smoothing of the empirical count/length distribution;
    # no duration parameters or word durations are fitted. This is intentionally
    # a weak prior: pauses and variation make duration ambiguous.
    bandwidth_frames = 4.0
    predictions, query_evidence = [], []
    for q in meta["queries"]:
        frames = raw[q["id"]].shape[0]
        length_mass = {
            n: float(np.exp(-0.5 * ((np.asarray(d) - frames) / bandwidth_frames) ** 2).sum())
            for n, d in frame_lengths.items()
        }
        ranked = []
        for candidate, pairs in support.items():
            n = len(candidate)
            score = length_mass[n] * len(pairs) / total_support[n]
            ranked.append((score, candidate))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        # Equal-length queries receive identical priors; IDs carry no evidence.
        chosen = ranked[0][1]
        predictions.append({"id": q["id"], "tokens": list(chosen)})
        total_score = sum(score for score, _ in ranked)
        alternatives = []
        for score, candidate in ranked[:10]:
            alternatives.append({
                "tokens": list(candidate), "unnormalized_prior_score": score,
                "normalized_heuristic_mass": score / total_score,
                "source_pair_count": len(support[candidate]),
                "splice_evidence": list(support[candidate].values())[:6],
            })
        query_evidence.append({"id": q["id"], "frames": frames,
                               "empirical_length_mass_by_token_count": length_mass,
                               "top_candidates": alternatives})

    method = (
        "Confirmed every query array is all zero. Made calibration-prior-only guesses: "
        "enumerated unseen whole-sentence prefix/suffix splices across identical calibration "
        "spans, counted supporting source pairs, and conditioned on recording length using "
        "the empirical calibration word-count/length distribution with a fixed four-frame "
        "Gaussian bandwidth. Ranked whole candidates deterministically; lexical ties are "
        "alphabetical. No neural-to-word model, word prototypes, or supervised projection. "
        "Reproduce with reconstruct_priors.py; evidence is in reconstruction_evidence.json."
    )
    limitations = [
        "Actual ordered query words are unidentifiable: all query neural activity is withheld.",
        "These are speculative prior-based complete sentences, not neural reconstructions; "
        "durations do not identify word counts, boundaries, identities, or order.",
        "Candidate support and normalized heuristic masses are not calibrated probabilities. "
        "The splice prior favors frequent calibration patterns and restricts lengths to the "
        "observed calibration range; alphabetical tie-breaking has no evidential meaning.",
        "Equal-length queries receive the same guess because their available evidence is identical.",
    ]
    result = {"predictions": predictions, "method": method, "limitations": limitations}
    evidence = {
        "query_checks": checks, "vocabulary": vocabulary,
        "calibration_count": len(calibration),
        "calibration_nonzero_values": sum(int(np.count_nonzero(raw[c["id"]])) for c in calibration),
        "candidate_count": len(support), "bandwidth_frames": bandwidth_frames,
        "calibration_frame_lengths_by_word_count": dict(frame_lengths),
        "candidate_support_totals_by_word_count": dict(total_support),
        "queries": query_evidence,
    }
    (root / "reconstruction_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (root / "predictions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
