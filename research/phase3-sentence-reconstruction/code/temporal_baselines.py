"""Shared word classifiers; calibration-only fitting, explicit development timing selection."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import itertools,json,sys,time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from sequence_study import ROOT,SEED,representation,save

def fit(a,y,shrink):
 return LinearDiscriminantAnalysis(solver='lsqr',shrinkage=shrink).fit(a.reshape(-1,a.shape[-1]),np.repeat(y[:,0],a.shape[1]))

def scores(model,b):
 return model.decision_function(b.reshape(-1,b.shape[-1])).reshape(len(b),b.shape[1])

def decode(s,cuts):
 return np.column_stack([(v.mean(1)>0).astype(int) for v in np.split(s,cuts,axis=1)])

def main():
 start=time.perf_counter();dev=np.load(ROOT/'cases/homogeneous_pc12_dev.npz');raw=np.load(ROOT/'data/sequence_features.npz')['frames']
 cal=raw[dev['train']];y=dev['ycal'];cv=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(cal,y[:,0]))
 results={}
 with threadpool_limits(limits=1):
  for components in [12,0]:
   shrinks=[.1,.5,.9,1.,'auto'];cv_losses=[]
   for shrink in shrinks:
    losses=[]
    for tr,va in cv:
     a,b,_=representation(cal[tr],cal[va],components)
     s=scores(fit(a,y[tr],shrink),b)
     # Calibration whole-utterance labels identify each frame's word here.
     losses.append(np.mean(np.logaddexp(0,-s*(y[va,0,None]*2-1))))
    cv_losses.append(float(np.mean(losses)))
   shrink=shrinks[int(np.argmin(cv_losses))]
   a,b,_=representation(cal,raw[dev['test']],components);model=fit(a,y,shrink);s=scores(model,b)
   candidates=[(a,b) for a in range(3,11) for b in range(a+3,17)]
   # This comparator gets twelve extra labeled mixed-order development trials
   # to choose timing; Sol's confirmation prompt never receives those labels.
   values=[int((decode(s,c)==dev['y']).all(1).sum()) for c in candidates]
   best=max(values);ties=[c for c,v in zip(candidates,values) if v==best]
   cuts=min(ties,key=lambda c:(abs(c[0]-20/3)+abs(c[1]-40/3),c))
   config={'components':components,'shrinkage':shrink,'cv_log_losses':cv_losses,'cut_candidates':candidates,'development_correct_by_cut':values,'selected_cuts':cuts,'development_n':len(b),'development_correct':best,'timing_selection':'Uses 12 extra labeled mixed-combination development recordings; favorable comparator, not strictly calibration-matched.'}
   results[f'pc{components}']=config
   for name in ['homogeneous_pc12_dev','homogeneous_pc12_confirm']:
    d=np.load(ROOT/'cases'/f'{name}.npz');_,b,_=representation(cal,raw[d['test']],components);s=scores(model,b)
    for label,ct in [('equal_thirds',(7,14)),('development_timed',cuts)]:
     pred=decode(s,ct);correct=pred==d['y']
     config[f'{name}_{label}']={'n':len(b),'exact_sequence_correct':int(correct.all(1).sum()),'exact_sequence_accuracy':float(correct.all(1).mean()),'token_accuracy':float(correct.mean()),'prediction_bits':pred.tolist(),'cuts':ct}
 results['wall_seconds']=time.perf_counter()-start
 save(ROOT/'evaluation/temporal_baselines.json',results)
 print(json.dumps({k:{z:v for z,v in x.items() if z in ['selected_cuts','shrinkage','development_correct'] or isinstance(v,dict) and 'exact_sequence_correct' in v} for k,x in results.items() if isinstance(x,dict)},default=str))

if __name__=='__main__':main()
