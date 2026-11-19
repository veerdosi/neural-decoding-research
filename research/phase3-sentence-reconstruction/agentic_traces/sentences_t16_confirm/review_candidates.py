from exemplar_decode import *
z=np.load('dtw_exemplar_scores.npz');M=json.load(open('exemplar_score_metadata.json'));words=M['words'];T=M['templates']
# Interpreted possibilities prompted by overlapping interval matches, not an LM.
candidates={
 'tb0b19ec8':['it','is','is it','it is','it music'],
 't59dcd856':['do you like music','you you like music','they do you like music','you like music'],
 't1fe83bc3':['i am good','i am good here','i am clean','i am here','i have it good','i have good'],
 'td28e3558':['you','you going','you are','you go','you it'],
 't6f0fdfe0':['i hope they do music','i hope they do not music','i hope they do not need music','i hope they do not like music','i hope they do not my music','i hope they do not need my music'],
 't4539ffbe':['i like it here','my not here','my is here','my glasses here','i am here','my here'],
 't514a6795':['my hope is my success','my hope is my success they'],
 't9ce92bac':['hungry','i hungry','am hungry','they it','my it','you hungry','hungry it'],
 't3afd73b4':['how thirsty are you','how hungry are you','how tired are you','my nurse are you','i nurse are do','nurse are you','how are you','my nurse are they','how nurse are you'],
 'tef9d94a6':['do they have my glasses','you do they have my glasses','do i have my glasses','do they my glasses','that do they my glasses','do they not my glasses'],
 't52b24bf6':['you hope family is good','you hope my family is good','you hope my family is that good'],
 't5f2d20a0':['my nurse is very tired','my nurse it very tired'],
 'te366b91c':['i hope you family','i hope you have family','i hope my family','i hope you my family'],
 'ta8a65775':['i hope need my do are','i hope they need my computer','i hope they need my music','i hope you need my computer','i hope need my computer','i hope they need my nurse','i hope they need my help here','i hope my computer','i hope they need my music here'],
 't41450074':['hello','it my','is my','i my','hello my','it'],
 't53cbb8eb':['i do not up','i do not have','i do not have it','i do not is','i do not need','i do not hope'],
 'tbfeb0ff2':['i hope family','i tell my family','i tell you family','i hope my family','i have family','i need family','i tell family','i tell my family hope'],
 'te3243945':['how tired are you','how tired are they','how are they','how are you'],
 'tca769afd':['is it you','it is you','it is do','is it do','it is hope you','is it my you','it my it do'],
 'td133fe68':['i like comfortable','i like comfortable family','i like my comfortable family','i like comfortable am','i am comfortable','i am comfortable am','i like help comfortable am','i like comfortable help','i like comfortable have','i like it comfortable'],
 't588781f6':['yes i very hungry','yes i am very hungry','you yes i very hungry','yes i am hungry','yes very hungry'],
 'tced8ecfc':['how do they do right','how they do right','what do they do right'],
 't6453e6ae':['how do do please','how do you feel please','how do you do please','how do please','how you do please','how do you clean please'],
 't4d2d7415':['it is hope you are comfortable','i hope you are comfortable','i do hope you are comfortable','yes i hope you are comfortable','it hope you are comfortable','is hope you are comfortable','i am hope you are comfortable']}
# All words fixed by each hypothesis, timing optimized using saved DTW scores.
# Start and end offsets can be unassigned; their fixed background score is zero.
result={}
for qid,cs in candidates.items():
 n=len(raw[qid]);spans=z[qid+'_spans'];W=z[qid+'_scores'];J=z[qid+'_template_index'];rr=[]
 for text in cs:
  toks=text.split()
  if any(w not in words for w in toks):continue
  D=np.full((len(toks)+1,n+1),-1e9);P={}
  for s in range(1,min(n-4,23)):D[0,s]=-.06*max(0,s-3)
  for k,w in enumerate(toks,1):
   wi=words.index(w)
   for si,(s,e) in enumerate(spans):
    val=D[k-1,s]+(float(W[si,wi])-.25)*(e-s)-.4
    if val>D[k,e]:D[k,e]=val;P[k,int(e)]=(int(s),si)
  e=max(range(max(5,n-14),n),key=lambda e:D[-1,e]);objective=float(D[-1,e]);path=[]
  for k in range(len(toks),0,-1):
   if (k,e) not in P:break
   s,si=P[k,e];wi=words.index(toks[k-1]);ti=int(J[si,wi]);path.append({'word':toks[k-1],'start':s,'end':e,'similarity':float(W[si,wi]),'exemplar':T[ti]});e=s
  rr.append({'tokens':toks,'objective':objective,'segments':path[::-1]})
 rr.sort(key=lambda r:-r['objective']);result[qid]=rr
 print('\n'+qid)
 for r in rr[:4]:print('%.3f '%r['objective']+' '.join(r['tokens'])+'  '+' '.join('%s:%d-%d/%.2f'%(s['word'],s['start'],s['end'],s['similarity']) for s in r['segments']))
Path('interpreted_candidate_comparisons.json').write_text(json.dumps(result,indent=2))
