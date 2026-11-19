from template_reconstruction import *
from scipy.ndimage import gaussian_filter1d
D=json.load(open('dtw_evidence.json'));C={c['id']:c for c in meta['calibration']};words=sorted(set(w for c in C.values() for w in c['tokens']));wi={w:i for i,w in enumerate(words)};out={};pred=[]
with open('dtw_votes_readable.txt','w') as f:
 for q in meta['queries']:
  qid=q['id'];n=q['shape'][0];votes=np.zeros((n,len(words),80));src={}
  for row in D[qid][4:]:
   for r in row['matches'][:8]:
    cid=r['id'];ci=int(cid[1:]);sim=r['similarity'];
    if cid not in B or sim<.25:continue
    for p,cp in enumerate(r['path']):
     qp=p+row['start'];ix=np.searchsorted(B[cid],cp,side='right')-1
     if ix<0 or ix>=len(C[cid]['tokens']):continue
     word=C[cid]['tokens'][ix];v=sim-.25
     if v>votes[qp,wi[word],ci]:votes[qp,wi[word],ci]=v
  score=np.sort(votes,axis=2)[:,:,-3:].sum(2)/3
  score=gaussian_filter1d(score,.8,axis=0)
  seq=np.argmax(score,axis=1);segments=[]
  for s in range(2,n-2):
   word=words[seq[s]]
   if not segments or segments[-1]['word']!=word:segments.append(dict(word=word,start=s,end=s+1))
   else:segments[-1]['end']=s+1
  tok=[s['word'] for s in segments if s['end']-s['start']>=3];pred.append(dict(id=qid,tokens=tok));out[qid]=dict(words=words,scores=score.tolist(),segments=segments)
  f.write('\n'+qid+' '+str([(s['word'],s['start'],s['end']) for s in segments])+'\n')
  for s in range(4,n-3,3):
   ix=np.argsort(score[s])[::-1][:4];f.write(str(s)+' '+', '.join(f'{words[j]}:{score[s,j]:.2f}' for j in ix)+'\n')
Path('dtw_vote_evidence.json').write_text(json.dumps(out));Path('algorithm_only_vote_predictions.json').write_text(json.dumps(pred,indent=2))
