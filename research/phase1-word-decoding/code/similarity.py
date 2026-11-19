"""Label-free signal-to-signal distances; Sol alone converts evidence to words."""
import hashlib, json
import numpy as np
from pilot import ROOT, WORDS, save

d = np.load(ROOT / 'evaluation' / 'prepared.npz')
train, test, donor = d['train'], d['test'], d['donor']
z, labels = d['z'], d['labels']
truth = json.loads((ROOT / 'evaluation' / 'ground_truth.json').read_text())
rng = np.random.default_rng(20261119)
reference_ids = [f'r{v:08x}' for v in rng.choice(2**32, len(train), replace=False)]

# This operation has no label argument. No voting, class averaging, learned
# metric, optimized neighborhood size, or signal-to-label prediction occurs.
def nearest_recordings(query_features, reference_features, k=12):
    distances = ((query_features[:, None, :] - reference_features[None, :, :])**2).mean(axis=2)
    ordering = np.argsort(distances, axis=1, kind='stable')[:, :k]
    return distances, ordering

header = '''You are the inference subject in a controlled neuroscience experiment.
Use only this prompt. Do not use tools, read files, browse, or delegate.
Identify the cued word for each independent neural recording. The participant
was instructed to imagine speaking a single word. Candidate words: ban, choice,
day, feel, kite, though, were. Query and reference IDs are random and contain no
information. Order is random. Do not infer label-count constraints or force
balanced outputs. Commit to one word per query.

Representation: each query is described by its distances to its 12 closest
individual calibration recordings. Distance is mean squared difference over
128 calibration-standardized neural features (64 threshold crossings and 64
log1p spike power, averaged 200-1200 ms after go). Smaller distances mean more
similar recordings. The distance calculation uses no labels and has no fitted
supervised model, class prototypes, votes, scores, or output predictions.
Each reference is one actual recording, never an average of a word class.
The reference table supplies individual word labels. Infer the word yourself
from these noisy similarities and the calibration labels, using whatever
evidence weighting you consider appropriate. No query metadata or labels are
supplied. Return ONLY JSON with predictions (list of objects with id and word,
every query exactly once) and method (short description of your decision rule
and main uncertainty).

Labeled calibration recordings:
'''
header += '\n'.join(json.dumps({'id': rid, 'word': WORDS[labels[t]]}, separators=(',', ':'))
                    for rid, t in zip(reference_ids, train))
for condition, trials in [('similarity_icl', test), ('similarity_shuffled', donor)]:
    distances, ordering = nearest_recordings(z[trials], z[train])
    prompt = header + '\n\nUnlabeled queries; neighbor pairs are [reference_id, distance]:\n'
    for t, row, dist in zip(truth, ordering, distances):
        neighbors = [[reference_ids[int(i)], round(float(dist[i]), 4)] for i in row]
        prompt += json.dumps({'id': t['id'], 'nearest_recordings': neighbors}, separators=(',', ':')) + '\n'
    path = ROOT / 'prompts' / f'{condition}.txt'
    path.write_text(prompt)
save(ROOT / 'evaluation' / 'similarity_config.json', {
    'representation': 'mean squared Euclidean distance over raw128 standardized features',
    'neighbors': 12, 'reference_n': len(train), 'labels_used_in_distance_calculation': False,
    'external_word_predictions_or_votes': False, 'normalization': 'identical to raw128_icl',
    'reference_ids_in_train_order': reference_ids,
    'prompts_sha256': {c: hashlib.sha256((ROOT/'prompts'/f'{c}.txt').read_bytes()).hexdigest()
                       for c in ['similarity_icl','similarity_shuffled']}})
print('Prepared two similarity prompts; no external signal-to-word predictor.')
