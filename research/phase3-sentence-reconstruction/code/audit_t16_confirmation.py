"""Reconstruct neural inputs and audit observable successful Codex actions."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,hashlib,json,pathlib,re
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,save
from sentence_study import words

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(name='sentences_t16_confirm'):
 case=json.loads((ROOT/'cases'/f'{name}.json').read_text());indices=np.load(ROOT/'cases'/f'{name}.npz');folder=resolve_sandbox(case['sandbox']);meta=json.loads((folder/'inputs.json').read_text());raw=np.load(folder/'neural_inputs.npz');source=WORK/'sources'/case['source'];assert sha(source)==case['source_sha256'];assert all(sha(folder/f)==h for f,h in case['input_sha256'].items())
 d=loadmat(source,variable_names=['binnedTX','spikePow','goTrialEpochs','blockNum','sentenceText','binSize'],simplify_cells=True);ch=np.array(case['channel_indices']);go=d['goTrialEpochs'].astype(int)-1;x=np.column_stack([d['binnedTX'][:,ch].astype(float),np.log1p(d['spikePow'][:,ch])])
 for b in np.unique(d['blockNum']):
  use=d['blockNum']==b;x[use]-=x[use].mean(0)
 def frame(i):
  a,b=go[i];s=x[a:b+1];return np.array([v.mean(0) for v in np.array_split(s,max(1,int(round(len(s)*float(d['binSize'])/100))))])
 arrays={int(i):frame(i) for i in np.r_[indices['train'],indices['donor']]};cal=np.concatenate([arrays[int(i)] for i in indices['train']]);mu=cal.mean(0);sd=cal.std(0,ddof=1);sd[sd<1e-8]=1;differences=[]
 for r,i in zip(meta['calibration'],indices['train']):
  assert r['tokens']==words(d['sentenceText'][i]);differences.append(float(abs(np.round((arrays[int(i)]-mu)/sd,3)-raw[r['id']]).max()))
 for r,i in zip(meta['queries'],indices['donor']):
  assert 'tokens' not in r;differences.append(float(abs(np.round((arrays[int(i)]-mu)/sd,3)-raw[r['id']]).max()))
 assert max(differences)==0
 pca=np.load(folder/'unsupervised_pca.npz');calraw=np.concatenate([raw[r['id']] for r in meta['calibration']]);pcamean=float(abs(calraw.mean(0)-pca['mean']).max());eigenvalues=np.linalg.eigvalsh(np.cov(calraw.T))[::-1];pcaerror=float(abs(eigenvalues-pca['eigenvalues']).max());assert pcamean<1e-10 and pcaerror<1e-8
 template_path=next((folder/f for f in ['calibration_word_intervals.json','calibration_interval_ledger.json','calibration_intervals.json'] if (folder/f).exists()),None);templates=json.loads(template_path.read_text()) if template_path else []
 if isinstance(templates,dict):templates=[{'id':cid,'word':t['token'],'start':t['start'],'end':t['end']} for cid,rows in templates.items() for t in rows]
 byid={r['id']:r for r in meta['calibration']};assert all(t['id'] in byid and t['word'] in byid[t['id']]['tokens'] and 0<=t['start']<t['end']<=len(raw[t['id']]) for t in templates)
 events=[json.loads(l) for l in (ROOT/'responses'/f'{name}.events.jsonl').read_text().splitlines()];commands=[e['item']['command'] for e in events if e.get('item',{}).get('type')=='command_execution' and e['type']=='item.completed'];flags=[]
 for c in commands:
  if re.search(r'loadmat|sentenceText|ground_truth|/evaluation/|https?://|curl |wget |requests\.|urllib',c):flags.append(c)
 save(ROOT/'evaluation'/('t16_confirmation_method_audit.json' if name=='sentences_t16_confirm' else name+'_method_audit.json'),{'case':name,'source_hash_matches':True,'input_hashes_match':True,'reconstructed_arrays':len(differences),'max_absolute_reconstruction_difference':max(differences),'calibration_only_PCA_mean_difference':pcamean,'calibration_only_PCA_eigenvalue_difference':pcaerror,'calibration_interval_exemplars':len(templates) if templates else None,'exemplar_word_types':len({t['word'] for t in templates}) if templates else None,'all_exemplars_from_calibration':True if templates else None,'reference_file':template_path.name if template_path else None,'word_boundaries':'Estimated by Sol from calibration overlaps, not observed linguistic boundaries. Comments calling these manual refer to model-written choices, not human annotations.','completed_command_count':len(commands),'suspicious_source_target_or_network_command_matches':flags,'commands':commands,'interpretation':'Sol constructed nonparametric word-labeled interval references and numerical scoring/assembly, then interpreted overlapping evidence. No pretrained neural decoder was supplied. Final inference is not tool-free neural reading.','audit_limits':'Observable commands and saved artifacts only; workspace-write is not an OS-level read-isolation guarantee. This is not proof about undisclosed model pretraining. Label-free full-block centering includes unexposed recordings.'});print({'case':name,'arrays':len(differences),'max_difference':max(differences),'templates_checked':len(templates),'flagged_commands':len(flags)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',default='sentences_t16_confirm');main(p.parse_args().case)
