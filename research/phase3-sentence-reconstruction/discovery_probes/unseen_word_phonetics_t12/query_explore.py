from models import *
B=np.load('whiten.npz');W=B['W'];V=gaussian_filter1d((Y-B['mu'])@W,1,axis=1);T=gaussian_filter1d((Q-B['mu'])@W,1,axis=1)
r=[]
for i,qid in enumerate(qids):
 d=((T-T[i])**2).sum((1,2)); c=((V-T[i])**2).sum((1,2));ii=np.argsort(d)[1:5];jj=np.argsort(c)[:8]
 item=dict(id=qid,queries=[(qids[j],round(float(d[j]),1)) for j in ii],calibration=[(words[j],round(float(c[j]),1)) for j in jj]);r.append(item);print(item)
json.dump(r,open('query_contrasts.json','w'),indent=2)
# noise expected query-query distance compared with calibration repeats
z=np.load('neural_inputs.npz');inp=json.load(open('inputs.json')); ds=[]
for w in words:
 trials=np.stack([gaussian_filter1d((z[x['id']]-B['mu'])@W,1,axis=0) for x in inp['calibration'] if x['word']==w]);ds.extend(pdist(trials.reshape(4,-1))**2 if False else [])
from scipy.spatial.distance import pdist
for w in words:
 trials=np.stack([gaussian_filter1d((z[x['id']]-B['mu'])@W,1,axis=0) for x in inp['calibration'] if x['word']==w]);ds.extend(pdist(trials.reshape(4,-1))**2)
print('same-word pair distance',np.quantile(ds,[0,.1,.5,.9,1]))
np.savez('query_whitened.npz',V=V,T=T)
