from retrieval import *
cs={c['id']:c for c in meta['calibration']};bounds=json.load(open('calibration_alignment.json'))['iterations'][-1];r=json.load(open('local_dtw_drop2.json'));out=[]
for q in meta['queries']:
 qid=q['id'];out.append('\n'+qid)
 for row in r[qid]:
  if row['end']-row['start']!=10:continue
  ss=[]
  for m in row['matches'][:5]:
   cid=m['id'];cuts=bounds.get(cid);ws=cs[cid]['tokens']
   if cuts:ws=[w for j,w in enumerate(ws) if min(cuts[j+1],m['end'])-max(cuts[j],m['start'])>1]
   ss.append(f"{m['similarity']:.2f} {cid}:{m['start']}-{m['end']} "+' '.join(ws))
  out.append(f"{row['start']}-{row['end']} "+' | '.join(ss))
Path('dtw_summary.txt').write_text('\n'.join(out))
