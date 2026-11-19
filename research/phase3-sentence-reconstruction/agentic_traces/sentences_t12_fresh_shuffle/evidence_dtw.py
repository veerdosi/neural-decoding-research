from local_dtw import *

def batch_scores(a,bs,penalty=.035):
    lens=np.array([len(b) for b in bs]);m=max(lens);n=len(a)
    B=np.zeros((len(bs),m,a.shape[1]))
    for k,b in enumerate(bs):B[k,:len(b)]=b
    S=np.einsum('nd,kmd->knm',a,B)
    D=np.full((len(bs),n+1,m+1),-1e9);D[:,0,:]=0
    for i in range(1,n+1):
        best=D[:,i-1,:m]+S[:,i-1,:]
        if i>=2:best=np.maximum(best,D[:,i-2,:m]+S[:,i-2,:]+S[:,i-1,:]-penalty)
        best[:,1:]=np.maximum(best[:,1:],D[:,i-1,:m-1]+.5*(S[:,i-1,:m-1]+S[:,i-1,1:])-penalty)
        D[:,i,1:]=best
    end=D[:,n,:];end[np.arange(m+1)[None,:]>lens[:,None]]=-1e9
    return np.max(end,axis=1)/n

if __name__=='__main__':
 A=make_arrays('drop2');cs=meta['calibration'];bs=[A[c['id']][2:-2] for c in cs];out={}
 for q in meta['queries']:
  qid=q['id'];a=A[qid];rows=[]
  for length in [6,10,16]:
   for start in range(2,len(a)-length-1,3):
    scores=batch_scores(a[start:start+length],bs);matches=[]
    for k in np.argsort(scores)[-12:][::-1]:
     r=subsequence(a[start:start+length],bs[k]);r['id']=cs[k]['id'];r['start']+=2;r['end']+=2
     r['path']=[(int(p[0]+start),int(p[1]+2)) for p in r['path']];matches.append(r)
    rows.append({'start':start,'end':start+length,'matches':matches})
  out[qid]=rows
 Path('local_dtw_drop2.json').write_text(json.dumps(out))
 with open('dtw_evidence.txt','w') as f:
  for qid,rows in out.items():
   print('\n###',qid,file=f)
   for row in rows:
    if row['end']-row['start']!=10:continue
    print(f"{row['start']:2}-{row['end']:2}",' | '.join(f"{r['similarity']:.3f} {r['id']}:{r['start']}-{r['end']} "+' '.join(cs[int(r['id'][1:])]['tokens']) for r in row['matches'][:5]),file=f)
 print('Wrote DTW evidence for',len(out),'queries')
