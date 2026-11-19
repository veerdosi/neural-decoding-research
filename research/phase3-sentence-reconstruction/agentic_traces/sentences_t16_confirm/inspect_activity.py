from local_retrieval import *
p=np.load('unsupervised_pca.npz');v=p['eigenvectors'];mu=p['mean'];out={}
for c in meta['calibration']+meta['queries']:
 x=gaussian_filter1d((raw[c['id']]-mu)@v[:,2:],1.2,axis=0);out[c['id']]=np.linalg.norm(x,axis=1).tolist()
Path('activity_norms.json').write_text(json.dumps(out))
for cid in ['c001','c026','c049','c073','tb0b19ec8','td28e3558','t9ce92bac','t41450074','t1fe83bc3','t4539ffbe','t514a6795','tca769afd']:
 print(cid,' '.join('%d:%.1f'%(i,x) for i,x in enumerate(out[cid]) if i%2==0))
