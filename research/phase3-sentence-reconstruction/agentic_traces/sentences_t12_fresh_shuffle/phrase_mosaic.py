from template_inference import *
if __name__=='__main__':
 A=make_arrays('drop2');x=json.load(open('calibration_interval_exemplars.json'));bounds=x['boundaries_history'][-1];dur=x['duration_frames'];refs=[];V=[]
 for c in meta['calibration']:
  b=bounds[c['id']];tok=c['tokens']
  for i in range(len(tok)):
   for j in range(i+1,min(i+5,len(tok))+1):
    s,e=b[i],b[j]
    if e-s<3:continue
    refs.append({'id':c['id'],'start':s,'end':e,'tokens':tok[i:j]});V.append(interval_vector(A[c['id']],s,e,10))
 V=np.array(V,np.float32);expected=np.array([sum(dur[w] for w in r['tokens']) for r in refs]);out={}
 for q in meta['queries']:
  qid=q['id'];T=len(A[qid]);spans=np.array([(s,e) for s in range(2,T-5) for e in range(s+3,min(s+50,T-3)+1)])
  Q=np.array([interval_vector(A[qid],s,e,10) for s,e in spans],np.float32);sims=Q@V.T
  lengths=spans[:,1]-spans[:,0]
  reward=lengths[:,None]*sims-.07*expected[None,:]*np.log(lengths[:,None]/expected[None,:])**2-.15
  # Prefer preserved within-recording phrase context by charging per retrieved span.
  reward[(lengths[:,None]/expected[None,:]<.45)|(lengths[:,None]/expected[None,:]>2.1)]=-1e6
  opts={end:[] for end in range(T)}
  for si,(s,e) in enumerate(spans):
   ix=np.argsort(reward[si])[-8:][::-1]
   for k in ix:opts[e].append((int(s),int(k),float(reward[si,k]),float(sims[si,k])))
  dp=[[] for _ in range(T)];dp[2]=[(0,(),[])]
  for e in range(5,T-2):
   candidates=[]
   for s,k,val,sim in opts[e]:
    for prev,toks,path in dp[s]:
     candidates.append((prev+val,toks+tuple(refs[k]['tokens']),path+[{'query_start':s,'query_end':e,'similarity':sim,'reference':refs[k]}]))
   candidates.sort(key=lambda x:-x[0]);seen=set()
   for v in candidates:
    if v[1] in seen:continue
    seen.add(v[1]);dp[e].append(v)
    if len(dp[e])==12:break
  out[qid]=[{'score':v[0],'tokens':v[1],'matches':v[2]} for v in dp[T-3]]
 Path('phrase_algorithm_only.json').write_text(json.dumps(out,indent=2))
 with open('phrase_evidence.txt','w') as f:
  for qid,rows in out.items():
   print('\n###',qid,file=f)
   for r in rows[:6]:
    print(f"{r['score']:.3f}",' '.join(r['tokens']),file=f)
    if r==rows[0]:
     for v in r['matches']:print(' ',v,file=f)
