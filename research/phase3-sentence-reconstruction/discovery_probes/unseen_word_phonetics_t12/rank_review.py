import numpy as np,json
from scipy.stats import rankdata
A=np.load('consensus_scores.npz');G=np.load('generative_scores.npz');Q=np.load('analysis_data.npz'); qids=list(Q['qids']);scores=A['scores'];ranks=np.stack([rankdata(scores[:,j],method='min') for j in range(16)],axis=1);granks=np.stack([rankdata(G['scores'][:,j],method='min') for j in range(16)],axis=1)
hyps=['camper','coupler','casbah','cowboy','coffee','copper','cucumber','computer','number','summer','slumber','camera','cupboard','couple','comber','culpa','subway','goodnight','tonight','quiet','guide','guided','gutter','aside','outside','aided','again','today','water','thirty','cheetah','shady','sugar','doctor','sister','daughter','thicker','dirty','tiger','shitty','city','outfit','upset','excite','outset','whiskey','icecream','iceskate','thankyou','thinking','thinker','picture','whisker','supine','cabbage','cable','cowbell','somebody','something','comfortable','turbine','sunshine','sunny','sunbeam','sunburn','cases','kisses','cousins','tissues','dishes','tennis','dozen','theaters','courteous','shyness','sinus','family','dismay','aflame','autism','telephone','asleep','happy','relic','retake','critic','rhetoric','music','drink','drinking','reading','writing','rabbit','graphic','traffic','garlic','erect','erratic','request','repeat','rescue','topic','chicken','awake','cheek','neck','medic','chronic']
for j,qid in enumerate(qids):
 r=[]
 for w in hyps:
  m=A['words']==w;g=G['words']==w
  if m.any():r.append((w,int(ranks[m,j].min()),int(granks[g,j].min()) if g.any() else 999999))
 r.sort(key=lambda a:a[1]);print(qid,r[:25])
