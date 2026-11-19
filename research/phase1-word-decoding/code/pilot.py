"""Focused, tool-free Codex neural word decoding pilot. No supervised decoder."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, subprocess, time, urllib.request
import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get('NEURAL_PILOT_WORK_DIR', str(ROOT.parents[1] / 'local'))).resolve()
WORDS = ['ban', 'choice', 'day', 'feel', 'kite', 'though', 'were']
MODEL = 'gpt-6.1-sol'
SEED = 20261118
HF = 'https://huggingface.co/datasets/raei/Kunz-2025-InnerSpeechMotorCortexImplicationsSpeechNeuroprostheses/resolve/main/'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def prepare():
    rng = np.random.default_rng(SEED)
    filename = 't12.2023.08.15_imagined1stMotor_raw.mat'
    source = WORK / 'sources' / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        urllib.request.urlretrieve(HF + 'isolatedVerbalBehaviors/' + filename, source)
    d = loadmat(source, simplify_cells=True)
    go = d['goTrialEpochs'].astype(int) - 1
    blocks = d['blockNum'][go[:, 0]].astype(int)
    labels = d['trialCues'].astype(int) - 2
    channels = np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')], dtype=int) - 1
    # Fixed 200-1200 ms after go onset. Never includes the displayed text period.
    start, stop = 200 // int(d['binSize']), 1200 // int(d['binSize'])
    tx = np.asarray(d['binnedTX'][:, channels], dtype=float)
    sp = np.log1p(np.asarray(d['spikePow'][:, channels], dtype=float))
    continuous = np.concatenate([tx, sp], axis=1)
    features = np.array([continuous[a+start:a+stop].mean(axis=0) for a in go[:, 0]])
    train = np.flatnonzero((labels >= 0) & np.isin(blocks, [8, 9]))
    test = np.flatnonzero((labels >= 0) & np.isin(blocks, [17, 25]))
    assert len(train) == len(test) == 98 and not set(train) & set(test)
    assert all(go[t, 0]+stop <= go[t, 1]+1 for t in np.r_[train, test])
    assert all(np.all(d['blockNum'][a+start:a+stop] == d['blockNum'][a]) for a in go[:, 0])
    # Center, scale, and PCA use calibration inputs only, without their labels.
    mu = features[train].mean(axis=0)
    scale = features[train].std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1
    z = (features - mu) / scale
    _, singular, vt = np.linalg.svd(z[train], full_matrices=False)
    pcs = vt[:12].copy()
    for v in pcs:
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    pca = z @ pcs.T
    train = rng.permutation(train)
    test = rng.permutation(test)
    ids = [f'q{v:08x}' for v in rng.choice(2**32, len(test), replace=False)]
    donor = test.copy()
    # Preserve the block distribution while breaking the recording-to-target link.
    for block in [17, 25]:
        idx = np.flatnonzero(blocks[test] == block)
        donor[idx] = rng.permutation(test[idx])
    np.savez_compressed(ROOT / 'evaluation' / 'prepared.npz', z=z, pca=pca,
        labels=labels, blocks=blocks, train=train, test=test, donor=donor,
        mu=mu, scale=scale, pcs=pcs, go=go)
    truth = [{'id': q, 'trial_index_zero_based': int(t), 'block': int(blocks[t]),
              'word': WORDS[labels[t]], 'shuffled_donor_trial': int(s),
              'shuffled_donor_word': WORDS[labels[s]]} for q, t, s in zip(ids, test, donor)]
    save(ROOT / 'evaluation' / 'ground_truth.json', truth)
    report = {
        'seed': SEED, 'model': MODEL, 'source_url': HF + 'isolatedVerbalBehaviors/' + filename,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'calibration_blocks': [8, 9], 'evaluation_blocks': [17, 25],
        'calibration_n': len(train), 'evaluation_n': len(test), 'words': WORDS,
        'array': 'i6v', 'channel_indices_zero_based': channels.tolist(),
        'window_ms_after_go': [200, 1200], 'features': '64 mean TX + 64 mean log1p spikePow',
        'standardization': 'calibration trial means only, ddof=1',
        'pca': '12 components from unlabeled calibration features, unwhitened',
        'pca_explained_variance_ratio': (singular[:12]**2 / np.sum(singular**2)).tolist(),
        'per_class_calibration': dict(collections.Counter(WORDS[labels[t]] for t in train)),
        'per_class_evaluation': dict(collections.Counter(WORDS[labels[t]] for t in test)),
        'exact_duplicate_train_test_features': sum(np.array_equal(features[t], features[s]) for t in train for s in test),
        'shuffled_same_word_count': int(sum(labels[t] == labels[s] for t, s in zip(test, donor))),
    }
    save(ROOT / 'config.json', report)
    base = (
        'You are the inference subject in a controlled neuroscience experiment. '
        'Use only information in this prompt. Do not use tools, read files, browse, or delegate. '
        'Identify the cued word for each independent neural recording. '
        'The participant was instructed to imagine speaking a single word. '
        'Candidate words: ' + ', '.join(WORDS) + '. '
        'Query IDs are random and contain no information. Query order is random. '
        'No metadata, timestamps, text stimuli, or evaluation labels are supplied. '
        'Do not infer any label-count constraint or force a balanced output. '
        'Infer participant-specific patterns from calibration examples when supplied. '
        'You may reason about distances, recurring dimensions, clusters, and noisy examples internally; '
        'do not call external tools or train a model. Commit to one candidate for every query. '
        'Return ONLY a JSON object with keys predictions and method. predictions must be a list '
        'of objects with id and word, with each query exactly once. method is a short description '
        'of your decision method and its main uncertainty.\n'
    )
    conditions = ['pca12_icl', 'raw128_icl', 'pca12_shuffled', 'raw128_shuffled', 'raw128_zero_shot', 'no_neural']
    for condition in conditions:
        raw = condition.startswith('raw128')
        matrix = z if raw else pca
        representation = ('128 features: first 64 are channelwise mean threshold-crossing counts; '
            'last 64 are channelwise mean log1p spike-band power. Each dimension is standardized '
            'using calibration recordings only. Channel ordering matches across examples.' if raw else
            '12 unwhitened principal-component coordinates, from 128 standardized channel features '
            '(64 threshold crossing and 64 log1p spike power). PCA uses only unlabeled calibration '
            'inputs. Coordinates match across examples; no coordinate has a semantic label.')
        prompt = base + '\nRepresentation: ' + representation + '\n'
        if condition == 'raw128_zero_shot':
            prompt += 'No labeled calibration examples are available. This is a zero-shot test.\n'
        else:
            prompt += '\nLabeled calibration examples, in random order:\n'
            for t in train:
                prompt += json.dumps({'word': WORDS[labels[t]], 'x': np.round(matrix[t], 2).tolist()}, separators=(',', ':')) + '\n'
        prompt += '\nUnlabeled queries:\n'
        actual = donor if condition.endswith('shuffled') else test
        for q, t in zip(ids, actual):
            row = {'id': q}
            if condition != 'no_neural':
                row['x'] = np.round(matrix[t], 2).tolist()
            prompt += json.dumps(row, separators=(',', ':')) + '\n'
        if condition == 'no_neural':
            prompt += '\nQuery neural recordings have been withheld. Still predict each word using whatever priors are available.\n'
        (ROOT / 'prompts' / f'{condition}.txt').write_text(prompt)
    print(json.dumps(report, indent=2))

def run(condition, effort='high'):
    cli = os.environ.get('CODEX_APP_RUNTIME_CLI_PATH') or 'codex'
    isolated = WORK / 'model_sandbox'
    isolated.mkdir(parents=True, exist_ok=True)
    prompt = ROOT / 'prompts' / f'{condition}.txt'
    response = ROOT / 'responses' / f'{condition}.json'
    command = [cli, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
        '--skip-git-repo-check', '-C', str(isolated), '-m', MODEL,
        '-c', f'model_reasoning_effort="{effort}"', '--json', '-o', str(response), '-']
    metadata = {'condition': condition, 'command': command, 'model': MODEL,
        'effort': effort, 'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    save(ROOT / 'responses' / f'{condition}.invocation.json', metadata)
    with prompt.open() as stdin, (ROOT / 'responses' / f'{condition}.events.jsonl').open('w') as stdout, \
            (ROOT / 'responses' / f'{condition}.stderr.txt').open('w') as stderr:
        try:
            result = subprocess.run(command, stdin=stdin, stdout=stdout, stderr=stderr, timeout=1200)
            metadata['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            metadata['returncode'] = 124
            metadata['error'] = 'Codex invocation exceeded 1200 seconds; no score is assigned.'
    metadata['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(ROOT / 'responses' / f'{condition}.invocation.json', metadata)
    print(condition, 'exit', metadata['returncode'], flush=True)

def parse_response(path):
    text = path.read_text().strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)

def evaluate():
    truth = json.loads((ROOT / 'evaluation' / 'ground_truth.json').read_text())
    words = np.array([t['word'] for t in truth])
    blocks = np.array([t['block'] for t in truth])
    rng = np.random.default_rng(SEED+1)
    results = {}
    predictions = {}
    for response in sorted((ROOT / 'responses').glob('*.json')):
        if response.name.endswith('.invocation.json'):
            continue
        invocation = response.with_suffix('.invocation.json')
        if not invocation.exists() or json.loads(invocation.read_text()).get('returncode') != 0:
            continue
        data = parse_response(response)
        pairs = data['predictions']
        assert len(pairs) == len(truth) and len(set(p['id'] for p in pairs)) == len(truth)
        lookup = {p['id']: p['word'] for p in pairs}
        assert set(lookup) == {t['id'] for t in truth} and set(lookup.values()) <= set(WORDS)
        pred = np.array([lookup[t['id']] for t in truth])
        predictions[response.stem] = pred
        correct = pred == words
        k, n = int(correct.sum()), len(correct)
        test_result = binomtest(k, n, 1/7, alternative='greater')
        ci = binomtest(k, n).proportion_ci(confidence_level=.95, method='exact')
        events = [json.loads(l) for l in response.with_suffix('.events.jsonl').read_text().splitlines() if l.strip()]
        # Any model-issued tool call invalidates this tool-free condition.
        items = [e['item'] for e in events if e.get('item')]
        invalid = [i.get('type') for i in items if i.get('type') != 'agent_message' and i.get('type') != 'reasoning']
        assert not invalid, (response.stem, invalid)
        # Block-restricted label permutation preserves imbalance, ID/output priors, and block effects.
        perm_counts = np.zeros(20000, dtype=int)
        for block in np.unique(blocks):
            idx = np.flatnonzero(blocks == block)
            perm = np.array([rng.permutation(words[idx]) for _ in range(20000)])
            perm_counts += (perm == pred[idx]).sum(axis=1)
        perm_p = float((1 + np.sum(perm_counts >= k)) / 20001)
        results[response.stem] = {
            'n': n, 'correct': k, 'accuracy': k/n, 'exact_binomial_95_ci': [ci.low, ci.high],
            'binomial_p_gt_chance_descriptive': test_result.pvalue,
            'block_restricted_permutation_p': perm_p,
            'block_accuracy': {str(b): float(correct[blocks == b].mean()) for b in np.unique(blocks)},
            'per_word_accuracy': {w: float(correct[words == w].mean()) for w in WORDS},
            'prediction_counts': dict(collections.Counter(pred.tolist())),
            'method': data.get('method'), 'tool_calls': len(invalid),
            'usage': [e.get('usage') for e in events if e['type'] == 'turn.completed'],
            'confusion_matrix_rows_truth_cols_pred': [[int(np.sum((words == a) & (pred == b))) for b in WORDS] for a in WORDS],
        }
        if response.stem.endswith('shuffled'):
            donor_words = np.array([t['shuffled_donor_word'] for t in truth])
            results[response.stem]['accuracy_against_actual_recording_donor_label'] = float(np.mean(pred == donor_words))
    comparisons = {}
    for real, control in [('pca12_icl', 'no_neural'), ('pca12_icl', 'pca12_shuffled'),
                          ('raw128_icl', 'no_neural'), ('raw128_icl', 'pca12_icl'),
                          ('raw128_icl', 'raw128_shuffled'),
                          ('similarity_icl', 'no_neural'), ('similarity_icl', 'similarity_shuffled'),
                          ('similarity_icl', 'raw128_icl')]:
        if real not in predictions or control not in predictions:
            continue
        a, b = predictions[real] == words, predictions[control] == words
        win, lose = int(np.sum(a & ~b)), int(np.sum(b & ~a))
        comparisons[real + '_vs_' + control] = {'accuracy_difference': float(a.mean()-b.mean()),
            'discordant_real_only_correct': win, 'discordant_control_only_correct': lose,
            'mcnemar_exact_one_sided_p_descriptive': binomtest(win, win+lose, .5, alternative='greater').pvalue if win+lose else 1.}
    for condition in ['pca12_icl', 'raw128_icl', 'similarity_icl']:
        if condition in results:
            results[condition]['chance_permutation_p_bonferroni_3_representations'] = min(1., 3 * results[condition]['block_restricted_permutation_p'])
    save(ROOT / 'evaluation' / 'metrics.json', {'conditions': results, 'comparisons': comparisons,
        'uncertainty_note': 'Binomial CIs and McNemar are descriptive: trials share recording blocks and model calls. Only two test blocks; no participant-level generalization claim.'})
    print(json.dumps({'conditions': {k:{a:v[a] for a in ['correct','n','accuracy','exact_binomial_95_ci','block_accuracy','block_restricted_permutation_p']} for k,v in results.items()}, 'comparisons': comparisons},indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'evaluate'])
    parser.add_argument('--condition')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run(args.condition)
    else: evaluate()
