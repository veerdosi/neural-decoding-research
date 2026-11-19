from local_dtw import *

def edge_match(a,b,penalty=.035):
 S=a@b.T;n,m=S.shape
 D=np.full((n+1,m+1),-1e9);D[0,0]=0
 for i in range(1,n+1):
  for j in range(1,m+1):
   cand=[D[i-1,j-1]+S[i-1,j-1]]
   cand.append(D[i-2,j-1]+S[i-2,j-1]+S[i-1,j-1]-penalty if i>=2 else -1e9)
   cand.append(D[i-1,j-2]+.5*(S[i-1,j-2]+S[i-1,j-1])-penalty if j>=2 else -1e9)
   D[i,j]=max(cand)
 j=np.argmax(D[n,:]);return float(D[n,j]/n),int(j)

if __name__=='__main__':
 import sys
 tag=sys.argv[1];A=make_arrays(tag);outs={}
 for q in meta['queries']:
  qid=q['id'];rows=[];print('\n',qid)
  for n in [5,8,12,16]:
   if len(A[qid])<n+4:continue
   matches=[]
   for c in meta['calibration']:
    sc,end=edge_match(A[qid][2:2+n],A[c['id']][2:-2]);matches.append({'id':c['id'],'similarity':sc,'end':end+2})
   matches.sort(key=lambda r:-r['similarity']);rows.append({'length':n,'matches':matches})
   print(n,' | '.join(f"{v['similarity']:.3f} {v['id']}:2-{v['end']} "+' '.join(meta['calibration'][int(v['id'][1:])]['tokens']) for v in matches[:6]))
  outs[qid]=rows
 Path('edge_'+tag+'.json').write_text(json.dumps(outs))
