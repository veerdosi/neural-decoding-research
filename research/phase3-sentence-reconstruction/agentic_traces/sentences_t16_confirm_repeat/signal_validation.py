from local_retrieval import *
led=json.load(open('calibration_interval_ledger.json'));p=np.load('unsupervised_pca.npz');mu=p['mean'];v=p['eigenvectors'];out={}
for tag in ['original','record_center','highpass']:
 A={}
 for k in raw.files:
  x=gaussian_filter1d((raw[k]-mu)@v,1.2,axis=0);x[:,:2]=0
  if tag=='record_center':x-=x.mean(0)
  if tag=='highpass':x-=gaussian_filter1d(x,6,axis=0)
  A[k]=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
 V=np.array([interval_vector(A[t['id']],t['start'],t['end']) for t in led]);S=V@V.T
 for i,t in enumerate(led):S[i,[j for j,x in enumerate(led) if x['id']==t['id']]]=-1
 ix=S.argmax(1);valid=[i for i,t in enumerate(led) if sum(x['word']==t['word'] for x in led)>1]
 acc=float(np.mean([led[i]['word']==led[ix[i]]['word'] for i in valid]));out[tag]=acc;print(tag,round(acc,3))
Path('signal_validation.json').write_text(json.dumps(out,indent=2))
