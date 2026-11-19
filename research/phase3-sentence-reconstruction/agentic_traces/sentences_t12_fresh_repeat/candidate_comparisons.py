"""Score explicitly interpreted candidate strings using only observed calibration
phrase intervals. Candidate text is human composition from saved neural retrieval;
rankings and timing paths are numerical evidence, not ground-truth labels.
"""
from fast_evidence import *
import sys
A=make_arrays('drop2');bs=json.load(open('interpreted_calibration_boundaries.json'))
H={
'tda989373':['they need what','they do what','they do that','they need that','they are what','they are that','they do not','they need it'],
'tdd25032b':['tell my family i do not feel thirsty','tell my family they do not feel thirsty','tell my family i do not feel very thirsty','tell my family i do not feel hungry','my family i do not feel thirsty'],
't5ee44a7f':['my music is here','bring music here','my music here','bring it here','my music is right here','my music they are here'],
'tdf8f9fae':['i hope they have','i hope you have','i hope i have','i hope they do','i hope they feel'],
't4f16b000':['goodbye have','goodbye music','goodbye hope','goodbye have faith','goodbye faith','goodbye please','goodbye coming','my family please','my family faith','my family','have faith','are they coming','are they here','my have faith','please have faith'],
't40a0a424':['no they do not feel very comfortable','no they do not feel comfortable','no i do not feel comfortable','no i do not feel very comfortable','they do not feel very comfortable','no they do not have comfortable','no they do not have very comfortable'],
'tf256b8ed':['tell my nurse is very good','tell my nurse i am very good','tell my nurse very good','my nurse is very good','tell my nurse i feel good','tell my nurse i feel very good','my nurse i am very good'],
't12be1627':['i like it','i like that','they like it','i bring up','they bring up','i like up','they like up','they need it','i like music','i like you','i like tired'],
't42cfafec':['i hope you have faith','i hope they have faith','i hope i have faith','i hope my faith','i hope you faith','i hope i faith'],
'te3715f88':['need my family good','need my family is good','tell my family good','tell my family is good','tell my family they are good','my family is good','you need my family good','tell my family very good','you need my family','tell my family are good'],
'tb8083852':['my family is not good','family is not good','they are not good','they do not good','it is not good','that is not good','my is not good','my nurse is not good','my music is not good','thirsty is not good','they do good'],
't9024802d':['do not need it','they do not need it','do not like it','do not like that','do not right here','you do not need it','they need it','do not need that','do not tell it'],
't2b08e042':['my faith is good','my faith is very good','my nurse is good','my nurse is very good','my faith are good','my faith they are good','my faith they good','my nurse are good'],
't62254eda':['that feel very comfortable','what feel very comfortable','they feel comfortable','that is very comfortable','what is very comfortable','they feel very comfortable','i feel very comfortable','that very comfortable','they very comfortable','that is my very comfortable','that my very comfortable'],
't9db27598':['is my nurse not comfortable','is my nurse comfortable','my nurse is not comfortable','you my nurse not comfortable','you nurse not comfortable','is my nurse very comfortable','you feel comfortable'],
'tef25d0d3':['how do not have faith','how do i have faith','they do not have my faith','they do not have faith','i do not have faith','you do not have faith','do not have faith','do you have faith','do you need my faith','they need my faith','do not need my faith','they do not need my faith','i do not need my faith','how do you have faith','they do not feel faith','how need not have faith']}
Path('interpreted_hypotheses.json').write_text(json.dumps(H,indent=2))
libph={}
for c in meta['calibration']:
 cid=c['id']
 if cid in ['c013','c034','c069']:continue
 for i in range(len(c['tokens'])):
  for j in range(i+1,min(i+4,len(c['tokens']))+1):
   key=tuple(c['tokens'][i:j]);s,e=bs[cid][i],bs[cid][j]
   libph.setdefault(key,[]).append((cid,s,e))
def gm(a,b):
 s=np.ascontiguousarray(a@b.T);r=np.zeros(4);p=np.zeros((len(a)+len(b),2),np.int32)
 lib.subdtw(s,len(a),len(b),.035,r,p,2)
 return float(r[0])
allout=json.load(open('candidate_comparison_scores.json')) if len(sys.argv)>1 else {}
for q in meta['queries']:
 cid=q['id']
 if len(sys.argv)>1 and cid not in sys.argv[1:]:continue
 a=A[cid];n=len(a);cache={};ranks=[]
 for sent in H[cid]:
  tokens=sent.split();K=len(tokens);D=np.full((K+1,n+1),-1e6);D[0,2]=0;P={}
  for k in range(K):
   for j in range(k+1,min(k+4,K)+1):
    key=tuple(tokens[k:j]);refs=libph.get(key,[])
    if not refs:continue
    for s in range(2,n-4):
     if D[k,s]<-1e5:continue
     for e in range(s+3,min(s+37,n-2)+1):
      ck=(key,s,e)
      if ck not in cache:
       scored=[]
       for rid,rs,re in refs:
        sim=gm(a[s:e],A[rid][rs:re])-.025*np.log((e-s)/(re-rs))**2
        scored.append((sim,rid,rs,re))
       sim,rid,rs,re=max(scored);cache[ck]=(sim*(e-s)-.20,dict(id=rid,start=rs,end=re,query_start=s,query_end=e,tokens=list(key),similarity=sim))
      val,ev=cache[ck];score=D[k,s]+val
      if score>D[j,e]:D[j,e]=score;P[j,e]=(k,s,ev)
  end=n-2;p=[];k=K
  while k and (k,end) in P:
   k,s,ev=P[k,end];p.append(ev);end=s
  ranks.append(dict(tokens=tokens,score=float(D[K,n-2]),matches=p[::-1]))
 ranks.sort(key=lambda r:-r['score']);allout[cid]=ranks
 print(cid,' | '.join(f"{r['score']:.2f} "+' '.join(r['tokens']) for r in ranks[:6]),flush=True)
Path('candidate_comparison_scores.json').write_text(json.dumps(allout,indent=2))
