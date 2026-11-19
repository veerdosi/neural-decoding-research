"""Paired, conditional and descriptive uncertainty analyses for one recording session."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,pathlib
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sequence_study import ROOT,SEED,save

def wilson(k,n):
 z=1.95996398454;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 return [float(center-half),float(center+half)]

def main():
 names=['homogeneous_pc12_confirm','homogeneous_pc12_confirm_repeat','homogeneous_pc12_confirm_shuffle','homogeneous_pc12_confirm_nosignal','homogeneous_pc12_confirm_reversed']
 available={n:json.loads((ROOT/'evaluation'/f'{n}.sol.json').read_text()) for n in names if (ROOT/'evaluation'/f'{n}.sol.json').exists()}
 case=np.load(ROOT/'cases/homogeneous_pc12_confirm.npz');y=case['y'];trial=case['test'];p=np.array(available[names[0]]['prediction_bits']);p2=np.array(available[names[1]]['prediction_bits']);correct=(p==y).all(1);correct2=(p2==y).all(1);counts=y.sum(1)
 rng=np.random.default_rng(SEED);n_perm=20000;null=np.zeros(n_perm);sort=np.argsort(trial);chunks=np.array_split(sort,3);groups=[chunk[counts[chunk]==count] for chunk in chunks for count in [1,2]]
 for i in range(n_perm):
  perm=np.arange(len(y))
  for idx in groups:perm[idx]=rng.permutation(idx)
  null[i]=.5*((p==y[perm]).all(1).sum()+(p2==y[perm]).all(1).sum())
 tb=json.loads((ROOT/'evaluation/temporal_baselines.json').read_text());b=np.array(tb['pc0']['homogeneous_pc12_confirm_development_timed']['prediction_bits']);bc=(b==y).all(1)
 pos_only=int(np.sum(correct&~bc));base_only=int(np.sum(~correct&bc));p_mcnemar=float(binomtest(min(pos_only,base_only),pos_only+base_only,.5).pvalue) if pos_only+base_only else 1.
 # Moving block bootstrap in chronological trial order; sensitivity, not a population CI.
 diffs=[];c=correct[sort].astype(float)-bc[sort].astype(float)
 for length in [5,10,20]:
  boot=[]
  for _ in range(10000):
   starts=rng.integers(0,len(c),int(np.ceil(len(c)/length)));idx=np.concatenate([(s+np.arange(length))%len(c) for s in starts])[:len(c)];boot.append(c[idx].mean())
  diffs.append({'block_length_trials':length,'paired_accuracy_difference_interval':np.quantile(boot,[.025,.975]).tolist()})
 result={'n_unique_confirmation_recordings':len(y),'n_independent_correct_signal_calls':2,'same_recordings_not_124_independent_trials':True,'sequence_agreement':float((p==p2).all(1).mean()),'correctness_agreement':float((correct==correct2).mean()),'descriptive_trial_wilson_interval':wilson(int(correct.sum()),len(y)),'interval_scope':'Within this session only; Wilson ignores serial dependence and is descriptive. No participant-population estimate.','bag_oracle_uniform_order_accuracy':1/3,'conditional_permutation':{'n':n_perm,'groups':'Word counts (bag) crossed with chronological thirds; same label permutation across both Sol calls','observed_mean_correct':float(.5*(correct.sum()+correct2.sum())),'null_mean_correct':float(null.mean()),'p_one_sided':float((1+(null>=.5*(correct.sum()+correct2.sum())).sum())/(1+n_perm))},'paired_raw_development_timed_lda':{'sol_correct':int(correct.sum()),'baseline_correct':int(bc.sum()),'sol_only_correct':pos_only,'baseline_only_correct':base_only,'mcnemar_two_sided_descriptive_p':p_mcnemar,'moving_block_bootstrap_sensitivity':diffs,'qualification':'Baseline gets extra mixed-order development labels for timing; 62 correlated same-session trials, no superiority claim.'},'by_sequence':[],'controls':{}}
 for code in range(1,7):
  idx=(y@np.array([4,2,1]))==code;result['by_sequence'].append({'tokens':['up' if v else 'right' for v in [(code>>2)&1,(code>>1)&1,code&1]],'n':int(idx.sum()),'correct':int(correct[idx].sum()),'repeat_correct':int(correct2[idx].sum())})
 for name,r in available.items():
  d=np.load(ROOT/'cases'/f'{name}.npz');pred=np.array(r['prediction_bits']);row={'exact_correct':r['exact_sequence_correct'],'word_correct':int((pred==d['y']).sum()),'n':len(pred),'donor_exact_correct':int((pred==d['donor_y']).all(1).sum())}
  if name.endswith('reversed'):row['reversed_truth_exact_correct']=int((pred==d['y'][:,::-1]).all(1).sum());row['reversed_truth_word_correct']=int((pred==d['y'][:,::-1]).sum());row['nonpalindrome_n']=int((d['y'][:,0]!=d['y'][:,-1]).sum());row['nonpalindrome_reversed_correct']=int(((pred==d['y'][:,::-1]).all(1)&(d['y'][:,0]!=d['y'][:,-1])).sum())
  result['controls'][name]=row
 save(ROOT/'evaluation/sequence_analysis.json',result)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,2,figsize=(12,4.8));labels=['Sol call 1','Sol call 2','Shuffle:\nnominal target','No signal','LDA + extra\ntiming labels'];values=[correct.sum(),correct2.sum(),available[names[2]]['exact_sequence_correct'],available[names[3]]['exact_sequence_correct'],bc.sum()]
 ax[0].bar(np.arange(len(labels)),np.array(values)/len(y),color=['#168e99','#168e99','#bd7045','#bd7045','#687a8b']);ax[0].set_xticks(np.arange(len(labels)),labels,rotation=20,ha='right');ax[0].set_ylim(0,1);ax[0].set_ylabel('Exact three-word accuracy');ax[0].axhline(1/3,ls='--',c='#777',lw=1,label='Known word bag, random order (1/3)');ax[0].legend(fontsize=8,loc='upper right')
 for i,v in enumerate(values):ax[0].text(i,v/len(y)+.025,f'{v}/62',ha='center',fontsize=10)
 per=result['by_sequence'];ax[1].bar(np.arange(6),[s['correct']/s['n'] for s in per],color='#168e99');ax[1].set_xticks(np.arange(6),[' '.join('U' if w=='up' else 'R' for w in s['tokens']) for s in per],rotation=25);ax[1].set_ylim(0,1.12);ax[1].set_ylabel('Exact sequence accuracy');ax[1].set_title('Every mixed order absent from calibration')
 for i,s in enumerate(per):ax[1].text(i,s['correct']/s['n']+.025,f"{s['correct']}/{s['n']}",ha='center')
 fig.suptitle('Frozen Sol reconstructs new word orders from 12 homogeneous examples',fontsize=13);fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/sequence_results.png',dpi=180);plt.close(fig)
 print(json.dumps({k:v for k,v in result.items() if k not in ['by_sequence','controls']},indent=2))

if __name__=='__main__':main()
