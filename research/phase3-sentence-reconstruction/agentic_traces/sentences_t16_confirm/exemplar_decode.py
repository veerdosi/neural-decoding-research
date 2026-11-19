"""Auditable direct interval retrieval; no neural model weights are fitted.
Manual calibration timing estimates follow calibration_overlap.txt.
Each word is represented by its original calibration trajectory, without averaging.
Query spans are compared using cosine similarity of eight interpolated time points.
Viterbi assembles unconstrained vocabulary tokens with a fixed per-word cost.
Outputs are algorithm-only hypotheses, to be interpreted against overlapping evidence.
"""
from local_retrieval import *
# Boundary list for each selected complete calibration transcript. Estimated from
# repeated-word/phrase neural overlaps, rounded to 100ms rows. No query labels used.
B={
0:[3,19,25,36,49,56,67],1:[2,8,18,30],2:[8,19,28,35,45,62],
3:[5,24,38,44,55],4:[3,15,23,35],5:[3,18,28,35,47,54,68],
6:[5,22,32,40,47,59],7:[3,20,29,35,44],8:[2,13,20,25,35,41,54],
9:[3,17,32,39,49,58],10:[2,19,25,42],11:[4,21,31,42,54],
12:[3,16,30,40,43,51,65],13:[5,17,25,34,53],14:[2,13,24,31,40],
15:[3,21,31,41,50,59],16:[4,22,29,46],17:[3,17,24,33],
18:[3,12,19,29],19:[3,14,23,32,35,46],21:[13,29,37,49],
22:[3,25,32,44,56],23:[3,16,26,31,43],24:[3,16,25,33,41,49,58],
25:[8,21,40],26:[2,17,31],27:[2,12,28,37,44,54],
28:[5,15,23,32,41,46,57],29:[3,20,23,31,42,52,60,67,81],
30:[3,22,29,35,46],31:[2,13,20,26,37],32:[4,25,33,46],
33:[3,22,28,33,48],34:[2,12,23,29,39],35:[3,17,25,32,42,49,58,64,77],
36:[2,17,31,38,50],38:[2,19,24,33,47],39:[9,20,27,35,43,51,64],
40:[5,18,25,31,43],41:[2,15,23,32,41],42:[3,13,21,33],
45:[2,13,22,24,35],46:[2,10,18,26,35,44,54],47:[3,25,33,41,54],
48:[3,18,30,34,40,46,58],49:[3,25,38],50:[3,25,29,39,47,56,62,69,83],
51:[2,12,22,32],52:[3,24,32,40,52],53:[2,20,34,47],54:[3,25,32,40,48,53,65],
55:[2,14,24,35,41,53],56:[2,23,31,39,48,61],57:[2,15,20,26,37],
58:[3,14,24,31,41,48,62],59:[2,23,31,41],60:[3,13,17,23,32,42],
61:[3,25,30,39,46,62],62:[2,10,19,23,32,47],63:[8,29,39,48,58,71],
64:[3,18,29,41],65:[3,23,29,43],66:[2,14,19,29,43],67:[6,18,27,34,41,55],
68:[4,20,29,38,46,60],69:[3,27,38,53],70:[2,16,27],
71:[3,18,22,31,41,51,56,74],72:[2,25,34,46],73:[2,14,29],
74:[2,14,25,30,40],75:[3,18,26,38],76:[3,20,27,36,39,52,72],
77:[3,24,33,41,52,63],78:[3,20,33,40,52,61,70,81],79:[2,19,25,31,45]}

def templates(A):
 T=[]
 for ci,bounds in B.items():
  c=meta['calibration'][ci];assert len(bounds)==len(c['tokens'])+1,(ci,bounds)
  for i,w in enumerate(c['tokens']):
   s,e=bounds[i:i+2]
   T.append({'word':w,'id':c['id'],'start':s,'end':e})
 Path('calibration_word_intervals.json').write_text(json.dumps(T,indent=2))
 return T

def main():
 A=make_arrays('drop2');T=templates(A);words=sorted(set(t['word'] for t in T));V=[];JT=[]
 for i,t in enumerate(T):
  # Small boundary shifts account for coarticulation and approximate annotation.
  for ds,de in [(0,0),(1,-1),(-1,1)]:
   s=max(2,t['start']+ds);e=min(len(A[t['id']])-2,t['end']+de)
   if e-s<4:continue
   V.append(interval_vector(A[t['id']],s,e));JT.append((i,s,e))
 V=np.asarray(V,np.float32)
 out={};npout={}
 for q in meta['queries']:
  qid=q['id'];a=A[qid];n=len(a);spans=[(s,e) for s in range(1,n-4) for e in range(s+4,min(n-1,s+33)+1)]
  Q=np.asarray([interval_vector(a,s,e) for s,e in spans],np.float32);S=Q@V.T
  W=np.full((len(spans),len(words)),-1.,np.float32);J=np.zeros_like(W,dtype=np.int32)
  for wi,w in enumerate(words):
   jj=[j for j,(ti,ss,ee) in enumerate(JT) if T[ti]['word']==w]
   vals=S[:,jj].copy()
   for j,j0 in enumerate(jj):
    ti,ss,ee=JT[j0];tl=ee-ss
    for si,(s,e) in enumerate(spans):
     ratio=(e-s)/tl
     if ratio<.42 or ratio>2.3:vals[si,j]=-1
   arg=vals.argmax(1);W[:,wi]=vals[np.arange(len(spans)),arg];J[:,wi]=np.asarray(jj)[arg]
  npout[qid+'_scores']=W;npout[qid+'_spans']=np.asarray(spans);npout[qid+'_template_index']=J
  # End margins may vary by two rows. No language-model or sentence prior.
  D=np.full(n+1,-1e9);prev={}
  for s in range(1,5):D[s]=-.1*abs(s-2)
  for si,(s,e) in enumerate(spans):
   wi=int(W[si].argmax());score=float(W[si,wi]);val=D[s]+(score-.25)*(e-s)-.55
   if val>D[e]:D[e]=val;prev[e]=(s,si,wi)
  end=max(range(n-5,n),key=lambda e:D[e]-.12*abs(e-(n-3)))
  path=[];ee=end
  while ee in prev:
   s,si,wi=prev[ee];ti,ts,te=JT[J[si,wi]]
   path.append({'word':words[wi],'start':s,'end':ee,'similarity':float(W[si,wi]),'exemplar':dict(T[ti],start=ts,end=te)});ee=s
  path=path[::-1]
  out[qid]={'tokens':[r['word'] for r in path],'segments':path,'objective':float(D[end])}
  print(qid,' '.join(out[qid]['tokens']),flush=True)
 Path('algorithm_only_predictions.json').write_text(json.dumps(out,indent=2));np.savez_compressed('exemplar_scores.npz',**npout)
 Path('exemplar_score_metadata.json').write_text(json.dumps({'words':words,'templates':T,'jittered_template_indices':JT,'settings':{'pca':'calibration only','remove_components':2,'smoothing_sigma':1.2,'interpolation_points':8,'word_cost':.55,'frame_baseline':.25}},indent=2))
if __name__=='__main__':main()
