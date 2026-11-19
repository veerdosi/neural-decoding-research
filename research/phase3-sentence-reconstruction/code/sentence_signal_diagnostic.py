"""Independent trained comparisons diagnose lexical information; never supplied to Sol."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,re,sys,gc
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from sequence_study import ROOT,WORK,save
from sentence_study import words

SOURCES={'t12_attempted':'t12.2023.11.28_attempted_50words.mat','t12_imagined':'t12.2023.11.28_imagined_50words.mat','t15_attempted':'t15.2024.03.03_attempted_50words.mat','t16_attempted':'t16.2024.04.29_attempted_50words.mat'}

def predict(a,b,y,alpha):
 mu=a.mean(0);std=a.std(0,ddof=1);std[std<1e-8]=1
 return Ridge(alpha=alpha).fit((a-mu)/std,y).predict((b-mu)/std)

def top(scores,k):
 result=np.zeros_like(scores,dtype=bool);idx=np.argsort(scores,axis=1)[:,-k:];np.put_along_axis(result,idx,True,axis=1);return result

def main(key):
 name=SOURCES[key];d=loadmat(WORK/'sources'/name,variable_names=['binnedTX','spikePow','goTrialEpochs','sentenceText','blockNum','chanSets','chanSetNames','decodedSentenceText'],simplify_cells=True);go=d['goTrialEpochs'].astype(int)-1;blocks=d['blockNum'][go[:,0]];unique=np.unique(blocks)
 ncal=3 if len(unique)>=5 else 2;cal=np.flatnonzero(np.isin(blocks,unique[:ncal]));dev=np.flatnonzero(blocks==unique[ncal])[:8]
 text=[words(s) for s in d['sentenceText']];vocab=sorted(set(w for i in cal for w in text[i]));lookup={w:i for i,w in enumerate(vocab)};target=np.zeros((len(text),len(vocab)),int)
 for i,t in enumerate(text):
  for w in t:
   if w in lookup:target[i,lookup[w]]=1
 features=[]
 for a,b in go:
  signal=np.column_stack([d['binnedTX'][a:b+1],np.log1p(d['spikePow'][a:b+1])]).astype(np.float32);features.append(np.array([p.mean(0) for p in np.array_split(signal,8)]).ravel())
 x=np.asarray(features);nch=d['binnedTX'].shape[1]
 for block in unique:x[blocks==block]-=x[blocks==block].mean(0)
 k=max(1,int(round(target[cal].sum(1).mean())));result={}
 for representation in ['all_arrays','ventral_array']:
  z=x
  if representation=='ventral_array':
   indices=np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')],int)-1;z=x.reshape(len(x),8,2*nch)[:,: ,np.r_[indices,indices+nch]].reshape(len(x),-1)
  alphas=[.1,1.,10.,100.,1000.];cv=[]
  for alpha in alphas:
   vals=[]
   for block in unique[:ncal]:
    tr=cal[blocks[cal]!=block];va=cal[blocks[cal]==block];s=predict(z[tr],z[va],target[tr],alpha);vals.append(f1_score(target[va],top(s,k),average='micro'))
   cv.append(float(np.mean(vals)))
  best=int(np.argmax(cv));s=predict(z[cal],z[dev],target[cal],alphas[best]);pred=top(s,k);prior=top(np.repeat(target[cal].mean(0)[None,:],len(dev),axis=0),k)
  result[representation]={'calibration_only_cv_micro_F1':cv,'selected_alpha':alphas[best],'bag_of_words_micro_F1':float(f1_score(target[dev],pred,average='micro')),'prior_only_micro_F1':float(f1_score(target[dev],prior,average='micro')),'top_k_from_calibration_only':k,'rows':[{'trial':int(i),'predicted_unordered_words':[w for w,b in zip(vocab,pred[j]) if b],'target_words':text[i]} for j,i in enumerate(dev)]}
 save(ROOT/'evaluation'/f'{key}.lexical_diagnostic.json',{'source':name,'calibration_n':len(cal),'development_n':len(dev),'calibration_blocks':unique[:ncal].tolist(),'development_block':int(unique[ncal]),'feedback_nonempty':int(sum(bool(np.size(v)) for v in d['decodedSentenceText'])),'normalization':'Unlabeled per-block feature means, then calibration-only scaling inside each CV fold','role':'Independent supervised bag-of-words diagnostic; no fitted maps/predictions enter Sol input','results':result})
 print(key,'cal',len(cal),{k:{z:v for z,v in val.items() if z in ['bag_of_words_micro_F1','prior_only_micro_F1','selected_alpha']} for k,val in result.items()},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):
  for key in sys.argv[1:]:main(key);gc.collect()
