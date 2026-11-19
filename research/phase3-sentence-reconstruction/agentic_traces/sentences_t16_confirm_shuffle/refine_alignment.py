"""Calibration-only, leave-one-recording-out template alignment.
Uses supplied transcript order only for calibration boundary refinement.
No weights, encoder, language model, or query transcript constraints are fitted.
"""
from template_reconstruction import *
from collections import Counter

def bank(A,bounds):
 T=[];V=[]
 for c in meta['calibration']:
  cid=c['id']
  if cid not in bounds:continue
  for i,w in enumerate(c['tokens']):
   s,e=bounds[cid][i:i+2];T.append(dict(word=w,id=cid,start=s,end=e,position=i));V.append(interval_vector(A[cid],s,e,12))
 return T,np.array(V,dtype=np.float32)

def interval_scores(a,T,V,exclude=None):
 n=len(a);spans=[(s,e) for s in range(1,n-3) for e in range(s+3,min(s+34,n-1)+1)]
 Q=np.array([interval_vector(a,s,e,12) for s,e in spans],np.float32);S=Q@V.T
 dur=np.array([t['end']-t['start'] for t in T]);ln=np.array([e-s for s,e in spans])
 S-= .035*np.log(ln[:,None]/dur[None,:])**2
 S[(ln[:,None]/dur[None,:]<.4)|(ln[:,None]/dur[None,:]>2.5)]=-1
 if exclude is not None:S[:,[i for i,t in enumerate(T) if t['id']==exclude]]=-1
 words=sorted(set(t['word'] for t in T));W=[];J=[]
 for w in words:
  ix=np.array([i for i,t in enumerate(T) if t['word']==w and (exclude is None or t['id']!=exclude)])
  if not len(ix):ix=np.array([i for i,t in enumerate(T) if t['word']==w])
  sc=S[:,ix];j=np.argmax(sc,axis=1);best=np.max(sc,axis=1)
  # Cross-instance support: best match plus half-weight second match.
  second=np.partition(sc,-min(2,len(ix)),axis=1)[:,-min(2,len(ix))]
  W.append((best+.5*second)/1.5);J.append(ix[j])
 return spans,words,np.array(W).T,np.array(J).T

def main():
 A=make_arrays('drop2');T,V=bank(A,B);bounds={};alignments={}
 for c in meta['calibration']:
  cid=c['id']
  if cid not in B:continue
  spans,words,W,J=interval_scores(A[cid],T,V,exclude=cid)
  lookup={v:i for i,v in enumerate(spans)};wi={w:i for i,w in enumerate(words)};n=len(A[cid]);tokens=c['tokens'];end=n-3
  dp=np.full((len(tokens)+1,n+1),-1e9);dp[0,2]=0;P={}
  for k,w in enumerate(tokens,1):
   for e in range(5,end+1):
    for s in range(max(2,e-34),e-2):
     ix=lookup.get((s,e))
     if ix is None:continue
     sc=dp[k-1,s]+W[ix,wi[w]]*(e-s)
     if sc>dp[k,e]:dp[k,e]=sc;P[k,e]=s
  if (len(tokens),end) not in P:bounds[cid]=B[cid];continue
  b=[end];e=end
  for k in range(len(tokens),0,-1):e=P[k,e];b.append(e)
  bounds[cid]=b[::-1]
 Path('refined_calibration_boundaries.json').write_text(json.dumps(bounds,indent=2))
 T,V=bank(A,bounds);out={};pred=[]
 for q in meta['queries']:
  qid=q['id'];a=A[qid];n=len(a);spans,words,W,J=interval_scores(a,T,V);lookup={v:i for i,v in enumerate(spans)}
  dp=np.full(n+1,-1e9);P={}
  # Up to 20 bins of leading idle allowed, costing .12 per skipped bin.
  for s in range(2,min(21,n-6)):dp[s]=-.12*(s-2)
  for e in range(5,n):
   for s in range(max(2,e-34),e-2):
    ix=lookup.get((s,e))
    if ix is None:continue
    j=int(np.argmax(W[ix]));sc=dp[s]+float(W[ix,j])*(e-s)-.45
    if sc>dp[e]:dp[e]=sc;P[e]=(s,j,ix)
  end=max(range(max(6,n-5),n),key=lambda e:dp[e]-.12*(n-3-e))
  seg=[];e=end
  while e in P:
   s,j,ix=P[e];order=np.argsort(W[ix])[::-1][:8]
   seg.append(dict(start=s,end=e,word=words[j],matches=[dict(word=words[k],similarity=float(W[ix,k]),template=T[J[ix,k]]) for k in order]));e=s
  seg=seg[::-1];tok=[v['word'] for v in seg];print(qid,' '.join(tok));pred.append(dict(id=qid,tokens=tok));out[qid]=dict(segments=seg)
 Path('algorithm_only_refined_predictions.json').write_text(json.dumps(pred,indent=2));Path('refined_template_evidence.json').write_text(json.dumps(out,indent=2))
 with open('refined_templates_readable.txt','w') as f:
  for qid,o in out.items():
   f.write('\n'+qid+'\n')
   for s in o['segments']:f.write(f"{s['start']}-{s['end']} "+' | '.join(f"{r['word']} {r['similarity']:.2f} {r['template']['id']}:{r['template']['start']}-{r['template']['end']}" for r in s['matches'][:5])+'\n')
if __name__=='__main__':main()
