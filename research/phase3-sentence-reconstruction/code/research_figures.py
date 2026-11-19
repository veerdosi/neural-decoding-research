"""Static research figures from saved predictions; no invented or pending results."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,save

def read(p):return json.loads(p.read_text())

def main():
 out=ROOT/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 symbolic=read(ROOT/'evaluation/neural_alphabet_confirmation_comparison.json');hmm=read(ROOT/'evaluation/neural_alphabet_hmm_comparison.json');full=read(ROOT/'evaluation/algorithm_sentence_baselines/full_evidence/t16_development_comparison.json')
 rows=[('Sol: symbols, no tools',100*read(ROOT/'evaluation/neural_alphabet_t16.sol.json')['WER'],100*symbolic['sol']['all']['WER'],'#d15c2c','-'),('HMM: identical symbols',100*hmm['WER'],100*symbolic['matched_HMM']['all']['WER'],'#888888','--'),('Numerical pipeline: full features',100*full['t16cal']['all_development8']['algorithm_wer'],100*symbolic['full_feature_contextual_comparisons']['t16cal']['all']['WER'],'#222222','-')]
 p=ROOT/'evaluation/t16_confirmation_comparison.json'
 if p.exists():rows.insert(0,('Sol: numerical tools',100*full['generic']['all_development8']['sol_wer'],100*read(p)['sol']['all']['WER'],'#d15c2c',':'))
 fig,ax=plt.subplots(figsize=(9,5.6));fig.subplots_adjust(left=.1,right=.65,bottom=.22,top=.88)
 for name,dev,test,color,style in rows:
  ax.plot([0,1],[dev,test],color=color,linestyle=style,marker='o',linewidth=2,label=name)
  ax.annotate(f'{test:.1f}%',(1,test),xytext=(9,0),textcoords='offset points',va='center',color=color,fontsize=10)
 ax.set_xlim(-.15,1.25);ax.set_ylim(0,85);ax.set_xticks([0,1],['Development\n8 recordings','Reserved\n32 recordings']);ax.set_ylabel('Word error rate (%) ↓');ax.grid(axis='y',color='#e6e6e6');ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(1.03,1),frameon=False,fontsize=9)
 fig.suptitle('Does the development result survive new recordings?',x=.1,ha='left',fontsize=16)
 fig.text(.1,.05,'T16 · 80 calibration sentences · one session / one evaluation block\nDifferent representations are labeled. Lines connect split summaries, not paired trials.',fontsize=9,color='#555555')
 for ext in ['png','svg']:fig.savefig(out/f't16_development_to_reserved.{ext}',dpi=220,facecolor='white')
 plt.close(fig);save(out/'t16_development_to_reserved.data.json',{'series':[{'method':n,'development_WER_percent':d,'reserved_WER_percent':t} for n,d,t,c,s in rows],'scope':'Single participant/session; development8 and reserved32 are distinct recordings in one block.'})
 records=[('Sol · symbols',symbolic['sol']['all'],'#d15c2c'),('HMM · same symbols',symbolic['matched_HMM']['all'],'#999999'),('Numerical pipeline · full features',symbolic['full_feature_contextual_comparisons']['t16cal']['all'],'#333333')]
 if p.exists():records.insert(0,('Sol · numerical tools',read(p)['sol']['all'],'#d15c2c'))
 fig,axes=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(left=.3,right=.94,top=.82,bottom=.2,wspace=.6);yy=np.arange(len(records))
 for ax,key,title in zip(axes,['WER','exact'],['Word error (%) ↓','Exact sentences /32 ↑']):
  vals=[100*r[key] if key=='WER' else r[key] for n,r,c in records];ax.barh(yy,vals,color=[c for n,r,c in records],height=.55);ax.invert_yaxis();ax.set_title(title,fontsize=12);ax.set_yticks(yy,[n for n,r,c in records] if ax is axes[0] else ['']*len(records));ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_xlim(0,80 if key=='WER' else 32)
  for y,v in zip(yy,vals):ax.text(v+(.8 if key=='WER' else .4),y,f'{v:.1f}' if key=='WER' else str(v),va='center',fontsize=11)
 fig.suptitle('Two measures, different conclusions',x=.03,ha='left',fontsize=16);fig.text(.03,.05,'Same32 reserved recordings. The full-feature numerical pipeline sees richer input than symbolic Sol.\n9 exact reconstructions are selected successes, not evidence of uniformly accurate language decoding.',fontsize=9,color='#555555')
 for ext in ['png','svg']:fig.savefig(out/f't16_reserved_comparison.{ext}',dpi=220,facecolor='white')
 plt.close(fig)
 # Illustrative successful recording, selected by a declared chronological rule.
 tool=read(ROOT/'evaluation/sentences_t16_confirm.sol.json');symby={r['id']:r for r in read(ROOT/'evaluation/neural_alphabet_t16_confirm.sol.json')['rows']};strict=set(read(ROOT/'evaluation/t16_confirmation_comparison.json')['strictly_novel_target_ids']);eligible=[r for r in tool['rows'] if r['id'] in strict and r['edit_distance']==0 and symby[r['id']]['edit_distance']==0 and len(r['truth'])>=6];r=min(eligible,key=lambda r:r['trial']);spec=read(ROOT/'cases/sentences_t16_confirm.json');d=loadmat(WORK/'sources'/spec['source'],variable_names=['binnedTX','goTrialEpochs','binSize'],simplify_cells=True);a,b=d['goTrialEpochs'][r['trial']].astype(int)-1;tx=d['binnedTX'][a:b+1][:,spec['channel_indices']];seconds=len(tx)*float(d['binSize'])/1000;codes=np.load(ROOT/'discovery_probes/neural_alphabet_t16_confirm/symbol_codes_for_analysis.npz')[r['id']]
 fig,axes=plt.subplots(2,1,figsize=(11,5.2),gridspec_kw={'height_ratios':[3,1]});fig.subplots_adjust(left=.09,right=.97,bottom=.24,top=.85,hspace=.28);axes[0].imshow(tx.T,aspect='auto',origin='lower',cmap='Greys',vmin=0,vmax=4,extent=[0,seconds,0,tx.shape[1]],interpolation='nearest');axes[0].set_ylabel('i6v channels');axes[0].set_title('Actual threshold-crossing counts · 20-ms bins · grayscale clips at 4',loc='left',fontsize=10)
 axes[1].scatter((np.arange(len(codes))+.5)*seconds/len(codes),codes,s=12,color='#d15c2c');axes[1].set_xlim(0,seconds);axes[1].set_ylim(-1,64);axes[1].set_yticks([0,31,63],['S00','S31','S63']);axes[1].set_ylabel('Cluster ID');axes[1].set_xlabel('Seconds after task go cue');axes[1].set_title('Calibration-only neural clusters · IDs are categorical, not a magnitude',loc='left',fontsize=10)
 fig.suptitle('A new sentence reconstructed from its recording',x=.09,ha='left',fontsize=16);fig.text(.09,.08,'SOL:       '+' '.join(r['prediction'])+'\nTARGET: '+' '.join(r['truth']),fontsize=12,color='#222222');fig.text(.09,.01,f'T16 · trial {r["trial"]+1} · selected successful held-out reconstruction. Text is not time-aligned to the signal.',fontsize=9,color='#555555')
 for ext in ['png','svg']:fig.savefig(out/f'actual_neural_sentence.{ext}',dpi=220,facecolor='white')
 plt.close(fig);save(out/'actual_neural_sentence.data.json',{'query_id':r['id'],'trial_zero_based':r['trial'],'source':spec['source'],'source_sha256':spec['source_sha256'],'channel_indices':spec['channel_indices'],'duration_seconds':seconds,'selection':'Earliest chronological strict-novel target with >=6 words decoded exactly by both tool-assisted and symbolic Sol. Illustrative success, not aggregate performance.','prediction':r['prediction'],'truth':r['truth'],'threshold_crossings':tx.tolist(),'cluster_codes':codes.tolist(),'display':'Grayscale counts clipped at4; cluster IDs categorical; no word timings supplied or inferred in this figure.'})

if __name__=='__main__':main()
