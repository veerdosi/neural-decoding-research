from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from sequence_study import ROOT,SEED,representation,save

def slots(x):return [v.mean(axis=1) for v in np.array_split(x,3,axis=1)]
def fit_predict(method,param,a,b,y):
 if method=='global_ridge':
  model=Ridge(alpha=param).fit(a.reshape(len(a),-1),y*2-1)
  return (model.predict(b.reshape(len(b),-1))>0).astype(int)
 if method=='global_lda':
  af=[a.reshape(len(a),-1)]*3;bf=[b.reshape(len(b),-1)]*3
 else:af=slots(a);bf=slots(b)
 pred=[]
 for pos in range(3):
  model=(LinearDiscriminantAnalysis(solver='lsqr',shrinkage=param) if 'lda' in method else KNeighborsClassifier(n_neighbors=param,weights='distance'))
  model.fit(af[pos],y[:,pos]);pred.append(model.predict(bf[pos]))
 return np.array(pred).T

def main(name):
 meta=json.loads((ROOT/'cases'/f'{name}.json').read_text());case=np.load(ROOT/'cases'/f'{name}.npz');data=np.load(ROOT/'data/sequence_features.npz')
 cal=data['frames'][case['train']];q=data['frames'][case['test']];y=case['ycal'];target=case['y'];classes=y@np.array([4,2,1])
 results={}
 grids={'slot_lda':[.1,.5,.9,1.,'auto'],'slot_knn':[1,3,5],'global_ridge':[.1,1.,10.,100.],'global_lda':[.1,.5,.9,1.,'auto']}
 cv=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(cal,classes))
 with threadpool_limits(limits=1):
  for components in [12,0]:
   a,b,_=representation(cal,q,components)
   for method,grid in grids.items():
    if method=='global_lda' and components==0:continue
    scores=[]
    for param in grid:
     fold=[]
     for tr,va in cv:
      x,z,_=representation(cal[tr],cal[va],components)
      pred=fit_predict(method,param,x,z,y[tr]);fold.append(np.mean((pred==y[va]).all(axis=1)))
     scores.append(float(np.mean(fold)))
    best=int(np.argmax(scores));pred=fit_predict(method,grid[best],a,b,y)
    result={'method':method,'components':components,'hyperparameter':grid[best],'calibration_cv_sequence_accuracies':scores,
      'prediction_bits':pred.tolist(),'exact_sequence_accuracy':float(np.mean((pred==target).all(axis=1))),
      'token_accuracy':float(np.mean(pred==target)),'position_accuracy':(pred==target).mean(axis=0).tolist(),
      'n':len(pred),'evaluation_role':meta['split'],'hyperparameter_selection':'Calibration data only, including scaler/PCA inside each fold'}
    results[f'{method}_pc{components}']=result
 save(ROOT/'evaluation'/f'{name}.baselines.json',results)
 print(name,{k:{'sequence':round(v['exact_sequence_accuracy'],3),'token':round(v['token_accuracy'],3),'position':np.round(v['position_accuracy'],2).tolist()} for k,v in results.items()},flush=True)

if __name__=='__main__':
 for name in sys.argv[1:]:main(name)
