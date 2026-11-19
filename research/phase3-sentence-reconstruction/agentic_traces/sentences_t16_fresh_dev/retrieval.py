import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from pathlib import Path

meta=json.load(open('inputs.json'))
raw=np.load('neural_inputs.npz')

def preprocess(x,sigma=0.8):
    x=gaussian_filter1d(x.astype(float),sigma,axis=0)
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)

arrays={k:preprocess(raw[k]) for k in raw.files}

def dtw(a,b):
    C=cdist(a,b,'cosine')
    n,m=C.shape
    D=np.full((n+1,m+1),np.inf); D[0,0]=0
    P=np.zeros((n,m),np.int8)
    for i in range(n):
        for j in range(m):
            v=[D[i,j],D[i,j+1]+0.12,D[i+1,j]+0.12]
            p=np.argmin(v)
            D[i+1,j+1]=C[i,j]+v[p]; P[i,j]=p
    i,j=n-1,m-1; path=[]
    while i>=0 and j>=0:
        path.append((i,j))
        p=P[i,j]
        if p==0:i-=1;j-=1
        elif p==1:i-=1
        else:j-=1
    path=path[::-1]
    return D[n,m]/len(path),path,C

if __name__=='__main__':
    out={}
    for q in meta['queries']:
        qid=q['id']; rows=[]
        for c in meta['calibration']:
            score,path,C=dtw(arrays[qid],arrays[c['id']])
            rows.append({'id':c['id'],'tokens':c['tokens'],'score':score,'path':path})
        rows.sort(key=lambda r:r['score']);out[qid]=rows
        print('\n',qid)
        for r in rows[:12]:print(round(r['score'],4),r['id'],' '.join(r['tokens']))
    Path('global_retrieval.json').write_text(json.dumps(out))
