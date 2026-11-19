from models import *
from variants import variant
y,q,tr=response();X=variant(P,5,.9,2);pairs=[('is','it'),('do','you'),('i','my'),('good','goodbye'),('feel','tell'),('that','what'),('no','need')];out=[]
for a,b in pairs:
 i,j=words.index(a),words.index(b);keep=~np.isin(np.arange(42),[i,j]);yp=fitpred(X[keep],y[keep],X[[i,j]],.3);pred=yp[1]-yp[0];actual=y[j]-y[i]
 out.append(dict(words=[a,b],pronunciations=[P[i],P[j]],both_word_types_excluded_from_fit=True,contrast_cosine=float(pred@actual/np.linalg.norm(pred)/np.linalg.norm(actual)),contrast_squared_error=float(np.sum((pred-actual)**2)),zero_contrast_squared_error=float(np.sum(actual**2))))
json.dump(out,open('calibration_contrast_evidence.json','w'),indent=2);print(out)
