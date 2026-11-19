"""A stronger shared-word LDA decoder with flexible monotone word boundaries."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import itertools,json
import numpy as np
from sequence_study import ROOT,representation,save
from temporal_baselines import fit,scores

PATTERNS=np.array(list(itertools.product([0,1],repeat=3)))
def decode(s,min_frames,penalty,expected):
 # Enumerate the two boundaries of three consecutive word states. All eight
 # strings are permitted; never constrain output to held-out mixed classes.
 total=s.shape[1];cuts=np.array([(a,b) for a in range(min_frames,total-2*min_frames+1) for b in range(a+min_frames,total-min_frames+1)])
 lengths=np.column_stack([cuts[:,0],cuts[:,1]-cuts[:,0],total-cuts[:,1]])
 duration_penalty=penalty*np.sum(((lengths-np.array(expected))/np.array(expected))**2,axis=1)
 cs=np.column_stack([np.zeros(len(s)),np.cumsum(s,axis=1)])
 segments=np.stack([cs[:,cuts[:,0]],cs[:,cuts[:,1]]-cs[:,cuts[:,0]],cs[:,-1,None]-cs[:,cuts[:,1]]],axis=-1)
 fit_score=np.einsum('nct,kt->nkc',segments,PATTERNS*2-1)-duration_penalty[None,None,:]
 best=fit_score.max(axis=2).argmax(axis=1)
 return PATTERNS[best]

def main():
 data=np.load(ROOT/'data/sequence_features.npz')['frames'];dev=np.load(ROOT/'cases/homogeneous_pc12_dev.npz');test=np.load(ROOT/'cases/homogeneous_pc12_confirm.npz');config=json.loads((ROOT/'evaluation/temporal_baselines.json').read_text());out={}
 for comp in [12,0]:
  conf=config[f'pc{comp}'];a,b,_=representation(data[dev['train']],data[dev['test']],comp);model=fit(a,dev['ycal'],conf['shrinkage']);sd=scores(model,b);_,q,_=representation(data[dev['train']],data[test['test']],comp);sq=scores(model,q)
  # Scale emission magnitudes by calibration only so a duration penalty has a
  # comparable meaning across feature sets. No query truth enters emissions.
  magnitude=np.std(scores(model,a));sd/=magnitude;sq/=magnitude
  cuts=conf['selected_cuts'];timing=[cuts[0],cuts[1]-cuts[0],20-cuts[1]]
  grid=[(n,p,e) for n in [1,2,3] for p in [0.,.1,.5,1.,4.] for e in [[20/3]*3,timing]]
  dev_correct=[int((decode(sd,*h)==dev['y']).all(1).sum()) for h in grid];chosen=int(np.argmax(dev_correct));pred=decode(sq,*grid[chosen]);fixed=decode(sq,2,.5,[20/3]*3)
  out[f'pc{comp}']={'shared_lda_shrinkage':conf['shrinkage'],'emission_fit':'12 homogeneous calibration recordings only','calibration_only_default':{'minimum_frames':2,'duration_penalty':.5,'expected_duration_frames':[20/3]*3,'exact_sequence_correct':int((fixed==test['y']).all(1).sum()),'prediction_bits':fixed.tolist()},'development_selected':{'selected_parameters':grid[chosen],'development_correct':dev_correct[chosen],'development_n':len(sd),'exact_sequence_correct':int((pred==test['y']).all(1).sum()),'n':len(pred),'word_correct':int((pred==test['y']).sum()),'prediction_bits':pred.tolist()},'grid':grid,'development_correct_by_parameter':dev_correct,'qualification':'Development selection gets 12 extra labeled mixed-order examples; not a matched calibration-only comparison.'}
 save(ROOT/'evaluation/segmented_baselines.json',out)
 print(json.dumps({k:{'default_correct':v['calibration_only_default']['exact_sequence_correct'],'selected':{z:w for z,w in v['development_selected'].items() if z!='prediction_bits'}} for k,v in out.items()},indent=2))

if __name__=='__main__':main()
