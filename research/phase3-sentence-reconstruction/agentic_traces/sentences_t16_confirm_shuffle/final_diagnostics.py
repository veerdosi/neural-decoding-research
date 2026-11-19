from template_reconstruction import *
from fast_evidence import match
A=make_arrays('drop2')
checks={
 'te3243945':[(2,21,[('c001',2,17,'i am'),('c051',2,22,'i need'),('c062',2,19,'i hope'),('c012',2,28,'i tell')])],
 'te366b91c':[(2,28,[('c062',2,19,'i hope'),('c035',2,24,'i hope'),('c012',2,40,'i tell you'),('c028',2,29,'i hope you'),('c012',2,28,'i tell')]),(10,18,[('c012',14,23,'tell'),('c002',17,27,'hope'),('c039',19,27,'hope'),('c028',13,22,'hope')]),(18,27,[('c024',25,33,'you'),('c066',14,21,'they'),('c074',23,30,'no'),('c028',13,22,'hope'),('c065',29,40,'okay')])],
 't1fe83bc3':[(10,18,[('c031',13,21,'do'),('c058',14,23,'do'),('c041',19,25,'you'),('c063',26,35,'you')])],
 't3afd73b4':[(2,22,[('c060',2,23,'yes i am'),('c059',2,30,'you are'),('c053',2,34,'how are'),('c020',2,38,'yes i am')])],
 'ta8a65775':[(23,37,[('c060',14,23,'i am'),('c048',29,41,'i am'),('c071',23,41,'you have'),('c024',25,41,'you are'),('c046',18,34,'they are')])],
 'td28e3558':[(2,16,[('c016',31,46,'hungry'),('c069',39,54,'hungry'),('c073',2,15,'my'),('c075',2,17,'they')]),(14,26,[('c014',32,39,'it'),('c073',15,30,'glasses'),('c061',45,62,'glasses'),('c039',50,64,'music')])],
 't848cf4da':[(2,23,[('c077',2,26,'i'),('c016',2,20,'i'),('c073',2,15,'my'),('c016',2,31,'i am'),('c026',2,31,'it is')])],
 'tf75891ff':[(21,31,[('c064',15,25,'need'),('c071',23,31,'you'),('c035',47,56,'need')]),(44,53,[('c072',24,34,'are'),('c053',22,34,'are'),('c057',27,36,'here')])],
 't4539ffbe':[(2,9,[('c041',2,10,'hope'),('c024',16,25,'hope'),('c078',18,30,'hope'),('c064',2,15,'they'),('c015',2,14,'they')]),(8,15,[('c066',14,21,'they'),('c050',38,46,'they'),('c064',15,25,'need'),('c075',17,26,'need')])],
 't588781f6':[(12,25,[('c050',46,61,'do not'),('c045',13,26,'like it'),('c024',41,49,'right'),('c048',41,50,'right'),('c050',53,61,'not')])],
}
out={}
for qid,qs in checks.items():
 print(qid);out[qid]=[]
 for s,e,refs in qs:
  rows=[]
  for cid,cs,ce,phrase in refs:
   r=match(A[qid][s:e],A[cid][cs:ce],2);rows.append(dict(tokens=phrase.split(),id=cid,start=cs,end=ce,similarity=r['similarity']))
  rows.sort(key=lambda r:-r['similarity']);out[qid].append(dict(start=s,end=e,matches=rows));print(s,e,' | '.join(f"{' '.join(r['tokens'])} {r['similarity']:.3f}" for r in rows))
Path('final_diagnostics.json').write_text(json.dumps(out,indent=2))
