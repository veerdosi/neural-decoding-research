import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pilot import ROOT

metrics = json.loads((ROOT/'evaluation/metrics.json').read_text())['conditions']
names = {'pca12_icl': 'PCA, in context', 'raw128_icl': '128 features, in context',
         'similarity_icl': 'Signal similarities, in context',
         'raw128_shuffled': '128 features, shuffled recordings',
         'pca12_shuffled': 'PCA, shuffled recordings', 'similarity_shuffled': 'Similarities, shuffled recordings',
         'raw128_zero_shot': '128 features, zero shot', 'no_neural': 'No query neural input'}
conditions = [c for c in names if c in metrics]
fig, ax = plt.subplots(figsize=(9.6, 5.3), layout='constrained')
for row, c in enumerate(conditions):
    m = metrics[c]
    color = '#2363a5' if c in ['pca12_icl','raw128_icl','similarity_icl'] else '#6c747a'
    lo, hi = np.array(m['exact_binomial_95_ci'])*100
    a = m['accuracy']*100
    ax.errorbar(a, row, xerr=[[a-lo],[hi-a]], fmt='o', color=color,
                ecolor=color, capsize=3, markersize=6, zorder=3)
    ax.text(max(hi+2, 29), row+.04, f"{m['correct']}/98 ({a:.1f}%)", fontsize=10, color=color)
ax.axvline(100/7, color='#a04f29', linestyle='--', linewidth=1.4, label='Chance: 14.3%')
ax.set_yticks(range(len(conditions)), [names[c] for c in conditions])
ax.invert_yaxis()
ax.set_xlim(0, max(72, max(metrics[c]['exact_binomial_95_ci'][1]*100 for c in conditions)+22))
ax.set_xlabel('Held-out word identification accuracy (%)')
ax.set_title('GPT-6.1 Sol through Codex: T12 motoric imagined speech', loc='left', fontsize=13, pad=18)
ax.grid(axis='x', alpha=.15)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.supxlabel('98 calibration trials; 98 evaluation trials in two later blocks.\nError bars: descriptive exact 95% trial-level intervals; one participant/session.', fontsize=9, color='#4f565a')
fig.savefig(ROOT/'evaluation/results.png', dpi=180, bbox_inches='tight', facecolor='white')
