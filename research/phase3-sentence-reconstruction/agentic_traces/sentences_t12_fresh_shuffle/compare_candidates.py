from template_inference import *
CANDIDATES={
 'tda989373':['bring my faith','goodbye my faith','is my faith','tell my faith','are my faith','is my nurse','my faith','have my faith','tell my nurse','my nurse please','is my family','tell my family','is my music','very good music'],
 'tdd25032b':['no i do not feel comfortable','no i do not feel very comfortable','no they do not feel comfortable','no i do not have comfortable','no i do not feel my comfortable','no i do not very comfortable'],
 't5ee44a7f':['my faith is good','my faith are good','my nurse is good','my faith is very good','my faith they good','my faith it good','my faith not good','my faith is going'],
 'tdf8f9fae':['they do not','they do that','they need that','they need it','they do what','they do tired','they are not','do they what','do they not','my i they what','they need help','they do it'],
 't4f16b000':['i like hope','i like that','i like it','i like help','i like up','i like tired','they like it','they like that','i like you','i like music','they like help','i need help'],
 't40a0a424':['tell my nurse not good','tell my nurse feel good','tell my nurse i am very good','tell my nurse i am good','tell my nurse they are good','tell my nurse you are good','tell my nurse is very good','tell my nurse very good','tell my nurse they feel good','tell my nurse i feel good','tell my nurse am very good','tell my nurse is good','tell my i need you good'],
 'tf256b8ed':['they do not have my faith','i do not have faith','i do not have my faith','you do not have faith','they do not have faith','i need my faith','they need my faith','i need not have faith','i do not feel faith','they do not feel faith','i need not feel faith','you need my faith'],
 't12be1627':['i hope they have it','i hope they have','i hope you have','i hope i have','i hope they do','i hope you do','i hope they are','i hope i am','i hope they feel','i hope they up','i hope you up'],
 't42cfafec':['they feel very comfortable','they are very comfortable','that is very comfortable','what is very comfortable','i feel very comfortable','you feel very comfortable','they very comfortable','that my very comfortable','is my very comfortable'],
 'te3715f88':['is my nurse comfortable','tell my nurse comfortable','you my nurse comfortable','is my nurse not comfortable','is my family comfortable','is my nurse comfortable you','my nurse is comfortable','is my nurse thirsty','is my faith comfortable','tell my nurse is comfortable'],
 'tb8083852':['do not need it','do not like it','do not like that','do not tell it','do not is it','do not feel it','they need it','they do not like it','do not that','do is that it','what is that it','do not it'],
 't9024802d':['my music is here','my music it here','bring music here','my music is it here','my music it is here','my music they here','my music is is here','my music right here','my music is right here'],
 't2b08e042':['my family is not good','my family do good','family is not good','family do good','my nurse is not good','my music is good','my family is good','my family not good','my here is not good','they do not feel good','my family do not good','my music is not good'],
 't62254eda':['tell my family goodbye','tell my family hello','tell my family not good','is my family very good','is my family going','tell my family good','tell my family is good','tell my family going','you need my family','i need my family','you need my family good','tell need my very good','is my family good','tell my family very good','you need my very good','tell need my family good','do you need my family'],
 't9db27598':['i hope i have faith','i hope you have faith','i hope they have faith','i hope i my faith','i hope you my faith','i hope do have faith','i hope my faith','they hope i have faith','i hope i have my faith','i hope you have my faith'],
 'tef25d0d3':['tell my family i do not feel thirsty','tell my family they do not feel thirsty','tell my family i do not feel very thirsty','tell my family i do not need feel thirsty','tell my family i do not they feel thirsty','tell my family i do not have feel thirsty','tell my family i do not feel hungry','tell my family i do not feel very hungry']}

def force(spans,scores,toks,T,dur):
 s,e=spans.T;L=e-s;D=np.full((len(toks)+1,T+1),-1e9);D[0,2]=0;P=np.full_like(D,-1,dtype=int)
 for j,w in enumerate(toks):
  ex=dur[WI[w]];v=L*scores[:,WI[w]]-.13*ex*np.log(L/ex)**2-.15
  vals=D[j,s]+v
  for k in np.argsort(vals):
   if vals[k]>D[j+1,e[k]]:D[j+1,e[k]]=vals[k];P[j+1,e[k]]=s[k]
 end=T-3;out=[]
 for j in range(len(toks),0,-1):
  start=int(P[j,end]);k=np.where((s==start)&(e==end))[0][0];out.append((toks[j-1],start,int(end),round(float(scores[k,WI[toks[j-1]]]),3)));end=start
 return float(D[len(toks),T-3]),out[::-1]

if __name__=='__main__':
 ss=np.load('query_interval_scores.npz');cc=np.load('consensus_scores.npz');dur=np.array([json.load(open('calibration_interval_exemplars.json'))['duration_frames'][w] for w in WORDS]);out={}
 for q in meta['queries']:
  qid=q['id'];spans=ss[qid+'_spans'];rawsc=ss[qid+'_scores'];cons=cc[qid+'_scores'];rows=[]
  for txt in CANDIDATES[qid]:
   toks=txt.split();sc,ali=force(spans,.5*rawsc+.5*cons,toks,q['shape'][0],dur)
   rows.append({'tokens':toks,'score':sc,'alignment':ali})
  rows.sort(key=lambda r:-r['score']);out[qid]=rows
 Path('candidate_comparison.json').write_text(json.dumps(out,indent=2))
 with open('candidate_comparison.txt','w') as f:
  for qid,rows in out.items():
   print('\n###',qid,file=f)
   for r in rows:print(round(r['score'],3),' '.join(r['tokens']),r['alignment'],file=f)
