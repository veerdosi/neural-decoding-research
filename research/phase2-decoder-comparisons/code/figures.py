import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import ROOT

E=ROOT/'evaluation';F=ROOT/'figures';F.mkdir(exist_ok=True)
summary=json.loads((E/'phase2_summary.json').read_text())
def sol(name):
 p=E/f'{name}.sol.json';return json.loads(p.read_text()) if p.exists() else None
def baseline(name):
 p=E/f'{name}.baseline.json';return json.loads(p.read_text()) if p.exists() else {}
colors={'Sol':'#215ca0','kNN':'#8b6c45','LDA':'#368166','GNB':'#9562a5'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained',sharey=True)
for ax,participant in zip(axes,['T12 development','T16 reserved participant']):
 ns=[2,4,7,14] if participant.startswith('T12') else [2,7]
 for label,key in [('Sol',None),('kNN','knn'),('LDA','lda_conservative_ties'),('GNB','gnb_conservative_ties')]:
  means=[];low=[];high=[];xs=[]
  for n in ns:
   if participant.startswith('T12'):
    row=summary['development_calibration_curve'][str(n)]
    values=row['sol_accuracies'] if label=='Sol' else row[key]['matched_draw_values']
   else:
    names=[f'confirm_raw_n{n}_draw{i}' for i in [0,1]]
    values=[s['accuracy'] for c in names if (s:=sol(c))] if label=='Sol' else [baseline(c)[key]['accuracy'] for c in names]
   if not values:continue
   xs.append(n);means.append(np.mean(values)*100);low.append(np.min(values)*100);high.append(np.max(values)*100)
  means=np.asarray(means)
  ax.errorbar(xs,means,yerr=[means-np.array(low),np.array(high)-means],color=colors[label],marker='o',capsize=3,label=label,linewidth=1.7)
 ax.axhline(100/7,color='#777',linestyle=':',linewidth=1);ax.set_title(participant,loc='left');ax.set_xticks(ns);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.13);ax.set_xlabel('Labeled calibration examples per word')
axes[0].set_ylabel('Accuracy on fixed later recordings (%)');axes[1].legend(frameon=False,loc='upper left')
fig.suptitle('Calibration efficiency: identical raw features and matched calibration subsets',fontsize=13)
fig.supxlabel('Bars show observed ranges, not population confidence intervals. T12: three Sol calls/draws (14 includes Phase 1). T16: two.\nBaseline curves use matched subsets; LDA/GNB use the conservative calibration-CV tie sensitivity.',fontsize=8)
fig.savefig(F/'calibration.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

rows=[('Raw post-go','dev_raw_n14_repeat1',['dev_raw_n14_repeat1','dev_raw_n14_repeat2']),
      ('Block centered','dev_blockcenter_n14_sol',['dev_blockcenter_n14_sol','dev_blockcenter_n14_sol_repeat']),
      ('Pre-go cue delay','dev_delay_n14_sol',['dev_delay_n14_sol']),
      ('Post minus pre-go','dev_delta_n14_sol',['dev_delta_n14_sol']),
      ('Top 12 PCs','dev_pca_n14_mechanism',[]),('Remaining PCs','dev_pca_tail_n14_mechanism',[]),
      ('Other cortical array','dev_other_n14_mechanism',[]),('Full orthogonal rotation','dev_rotation_n14',['dev_rotation_n14'])]
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for i,(label,bcase,scases) in enumerate(rows):
 b=baseline(bcase);vals=[s['accuracy'] for c in scases if (s:=sol(c))]
 if label=='Raw post-go':vals=[52/98]+vals
 if label=='Top 12 PCs':vals=[31/98]
 if vals:
  mean=np.mean(vals)*100;ax.errorbar(mean,i-.13,xerr=[[mean-min(vals)*100],[max(vals)*100-mean]],fmt='o',color=colors['Sol'],capsize=3,label='Sol' if i==0 else None)
 key='lda'
 if key in b:ax.scatter(b[key]['accuracy']*100,i+.13,marker='s',color=colors['LDA'],label='Regularized LDA' if i==0 else None)
ax.set_yticks(range(len(rows)),[r[0] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,100);ax.axvline(100/7,color='#777',linestyle=':');ax.grid(axis='x',alpha=.15);ax.set_xlabel('Development accuracy (%)');ax.set_title('Which transformations change performance?',loc='left');ax.legend(frameon=False,loc='lower right')
fig.supxlabel('All rows use the same T12 evaluation recordings; preprocessing is calibration-only except explicitly transductive block centering.\nSol dots are call means with observed ranges; most mechanism probes have one call. PCA Sol result is from Phase 1.',fontsize=8)
fig.savefig(F/'mechanisms.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

fig,ax=plt.subplots(figsize=(9.7,4.5),layout='constrained')
names=[('Seven labels, 2 examples','confirm_raw_n2_draw0'),('Seven labels, 7 examples','confirm_raw_n7_draw0'),
       ('Shuffled recordings','confirm_raw_n7_shuffled'),('No query signal','confirm_raw_n7_no_signal'),('Block-centered features','confirm_blockcenter_n7')]
for row,(label,name) in enumerate(names):
 s=sol(name)
 if not s:continue
 values=[s['accuracy']]
 if name in ['confirm_raw_n2_draw0','confirm_raw_n7_draw0']:
  extra=sol(name[:-1]+'1')
  if extra:values.append(extra['accuracy'])
 ax.scatter(np.asarray(values)*100,np.full(len(values),row),color=colors['Sol'],zorder=3)
 ax.text(max(values)*100+2,row+.05,', '.join(f'{v*100:.1f}%' for v in values),fontsize=10)
ax.set_yticks(range(len(names)),[n[0] for n in names]);ax.invert_yaxis();ax.set_xlim(0,100);ax.axvline(100/7,color='#777',linestyle=':',label='Chance 14.3%');ax.set_xlabel('Accuracy on the reserved 49 T16 recordings (%)');ax.set_title('Replication in a second participant',loc='left');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower right')
fig.supxlabel('Participant-specific calibration in block 9; evaluation in later block 17. Repeated calls reuse the same 49 recordings.\nProtocol and secondary block-centering condition were frozen before the first T16 evaluation score.',fontsize=8)
fig.savefig(F/'confirmation.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
print('Saved calibration, mechanism and confirmation figures.')
