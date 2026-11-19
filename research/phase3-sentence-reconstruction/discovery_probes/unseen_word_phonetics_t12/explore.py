import json,re,numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import svd
inp=json.load(open('inputs.json')); z=np.load('neural_inputs.npz')
words=sorted(set(x['word'] for x in inp['calibration']))
D={}
for line in open('cmudict.dict'):
 a=line.split()
 if len(a)<2 or a[0].startswith(';;;'):continue
 w=re.sub(r'\(\d+\)$','',a[0]).lower(); p=tuple(re.sub('[012]','',x) for x in a[1:])
 D.setdefault(w,[])
 if p not in D[w]:D[w].append(p)
P=[D[w][0] for w in words]
Y=np.stack([np.mean([z[x['id']] for x in inp['calibration'] if x['word']==w],axis=0) for w in words])
allY=np.stack([z[x['id']] for x in inp['calibration']]); rep=np.stack([z[x['id']]-Y[words.index(x['word'])] for x in inp['calibration']])
print('words',[(w,' '.join(p)) for w,p in zip(words,P)])
print('Y std',Y.std(),'within std',rep.std(),'time energy',np.round(np.mean(Y**2,axis=(0,2)),2))
s=svd(Y.reshape(42,-1)-Y.reshape(42,-1).mean(0),compute_uv=False);print('word singular',np.round(s[:20],2))
s=svd(Y.reshape(-1,256)-Y.reshape(-1,256).mean(0),compute_uv=False);print('channel singular',np.round(s[:30],2))
phones=sorted(set(sum(P,()))); B=np.array([[p.count(ph) for ph in phones] for p in P]); C=np.corrcoef(squareform(pdist(B))[np.triu_indices(42,1)],squareform(pdist(Y.reshape(42,-1)))[np.triu_indices(42,1)])[0,1]
print('distance correlation phoneme counts',C)
np.savez('analysis_data.npz',Y=Y,words=words,P=np.array([' '.join(p) for p in P]),Q=np.stack([z[x['id']] for x in inp['queries']]),qids=[x['id'] for x in inp['queries']])
json.dump(D,open('pronunciations.json','w'))
