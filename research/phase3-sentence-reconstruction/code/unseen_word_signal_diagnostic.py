"""Independent known-word diagnostic for the unseen-lexical-type source.

Not an unseen-word comparator: asks whether simple decoders retain word
information across the same calibration/evaluation recording blocks.
"""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from sequence_study import ROOT,WORK,save

def main():
 gt=json.loads((ROOT/'discovery_probes/unseen_word_phonetics_t12_ground_truth.json').read_text());d=loadmat(gt['source'],variable_names=['tx1','spikePow','blockNum','goTrialEpochs','trialCues','cueList'],simplify_cells=True);go=d['goTrialEpochs'].astype(int)-1;blocks=d['blockNum'][go[:,0]];labels=np.array([str(d['cueList'][int(i)-1]).lower() for i in d['trialCues']]);train=np.array(gt['calibration_trials']);calwords=set(labels[train]);test=np.array([i for i,w in enumerate(labels) if blocks[i]==18 and w in calwords]);assert len(test)==84
 x=np.column_stack([d['tx1'][:,:128].astype(float),np.log1p(d['spikePow'][:,:128])])
 for b in np.unique(d['blockNum']):
  ids=d['blockNum']==b;x[ids]-=x[ids].mean(0)
 def feature(i):
  a,b=go[i];return np.array([v.mean(0) for v in np.array_split(x[a:b+1],20)])
 tr=np.stack([feature(i) for i in train]);te=np.stack([feature(i) for i in test]);mu=tr.reshape(-1,256).mean(0);sd=tr.reshape(-1,256).std(0,ddof=1);sd[sd<1e-8]=1;tr=((tr-mu)/sd).reshape(len(tr),-1);te=((te-mu)/sd).reshape(len(te),-1)
 models={'kNN3_full_trajectory':KNeighborsClassifier(3,metric='cosine')};pca=PCA(64,svd_solver='full').fit(tr);models['shrinkage_LDA_PCA64']=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto');results={}
 for name,model in models.items():
  a,b=(pca.transform(tr),pca.transform(te)) if 'PCA' in name else (tr,te);model.fit(a,labels[train]);pred=model.predict(b);results[name]={'correct':int((pred==labels[test]).sum()),'n':len(test),'accuracy':float((pred==labels[test]).mean()),'predictions':[{'trial':int(i),'truth':str(t),'prediction':str(p)} for i,t,p in zip(test,labels[test],pred)]}
 save(ROOT/'evaluation/unseen_word_source_known_word_diagnostic.json',{'scope':'Independent known-word diagnostic, not a fair unseen-word decoding comparator. No outputs provided to Sol. Model settings fixed before evaluating these84 additional recordings. Same block centering, calibration-only scaling, arrays and100-ms frames as novel-word probe.','calibration_n':len(train),'known_word_test_n':len(test),'chance_accuracy':1/len(calwords),'calibration_blocks':[2,4],'test_block':18,'PCA64_calibration_variance_retained':float(pca.explained_variance_ratio_.sum()),'results':results});print(json.dumps({k:{a:b for a,b in v.items() if a!='predictions'} for k,v in results.items()},indent=2))

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
