"""Verify v2 media, source recordings, prediction provenance and reveal order."""
import hashlib,json,pathlib,re,subprocess
import numpy as np
from scipy.io import loadmat
from PIL import Image,ImageDraw

HERE=pathlib.Path(__file__).resolve().parent;MEDIA=HERE.parent;PROJECT=MEDIA.parent;ROOT=PROJECT/'research/phase3-sentence-reconstruction'
import sys
sys.path.insert(0,str(ROOT/'code'))
from paths import resolve_sandbox
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run(args):return subprocess.run(args,text=True,capture_output=True,check=True)

def main():
 m=read(HERE/'public/evidence.json');assert len(m['shots'])==4;assert [len(s['prediction']) for s in m['shots']]==[3,5,6,8];checks=[];loaded={}
 for s in m['shots']:
  for name,h in s['evidence_hashes'].items():assert sha(ROOT/name)==h
  case=read(ROOT/'cases'/f"{s['source_case']}.json");idx=np.load(ROOT/'cases'/f"{s['source_case']}.npz");qi=case['query_ids'].index(s['query_id']);assert s['trial_zero_based']==int(idx['test'][qi])==int(idx['donor'][qi]);assert s['trial_zero_based'] not in idx['train']
  for name in [s['source_case'],s['repeat_case']]:
   r=next(x for x in read(ROOT/'evaluation'/f'{name}.sol.json')['rows'] if x['id']==s['query_id']);assert r['prediction']==r['truth']==s['prediction']==s['target']
  source=PROJECT/'local/sources'/s['source_file'];assert sha(source)==s['source_sha256']
  if str(source) not in loaded:loaded[str(source)]=loadmat(source,variable_names=['binnedTX','goTrialEpochs'],simplify_cells=True)
  d=loaded[str(source)];a,b=d['goTrialEpochs'][s['trial_zero_based']].astype(int)-1;raw=d['binnedTX'][a:b+1][:,case['channel_indices'][:64]];sig=read(HERE/'public'/s['signal_asset']);assert np.array_equal(raw,np.asarray(sig['counts']));assert sha(HERE/'public'/s['signal_asset'])==s['signal_sha256']
  expected=np.round(np.clip(raw.T/4,0,1)*255).astype('uint8')
  for raster in [s['dark_raster'],s['light_raster']]:assert np.array_equal(np.array(Image.open(HERE/'public'/raster))[:,:,3],expected)
  assert s['word_start_seconds']>s['signal_seconds'];assert s['word_start_seconds']+(len(s['prediction'])-1)*s['word_step_seconds']+.22<s['truth_start_seconds'];assert s['truth_start_seconds']+.48<s['duration_seconds']
  checks.append({'query_id':s['query_id'],'native_signal_exact':True,'prediction_and_repeat_exact':True,'calibration_disjoint':True,'prediction_precedes_target':True})
 movie=MEDIA/'video_v2.mp4';probe=json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(movie)]).stdout);v=next(x for x in probe['streams'] if x['codec_type']=='video');audio=next(x for x in probe['streams'] if x['codec_type']=='audio');assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'30/1',960);assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and audio['codec_name']=='aac';assert abs(float(probe['format']['duration'])-32)<.05
 decode=run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-']);assert not decode.stderr.strip()
 volume=run(['ffmpeg','-hide_banner','-i',str(movie),'-af','volumedetect','-vn','-sn','-dn','-f','null','-']).stderr;peak=float(re.search(r'max_volume:\s*([\-\d.]+)',volume).group(1));assert -60<peak<-1
 frames=[0,45,90,135,210,300,360,435,525,600,675,780,855,930];out=HERE/'encoded-frames';out.mkdir(exist_ok=True);selection='+'.join(f'eq(n\\,{i})' for i in frames)
 run(['ffmpeg','-v','error','-y','-i',str(movie),'-vf','select='+selection,'-fps_mode','vfr',str(out/'frame-%02d.png')]);images=sorted(out.glob('*.png'));assert len(images)==len(frames)
 w,h=640,360;sheet=Image.new('RGB',(w*2,(h+28)*7),'#292a27');draw=ImageDraw.Draw(sheet)
 for j,(p,f) in enumerate(zip(images,frames)):
  im=Image.open(p).convert('RGB');im.thumbnail((w,h));x=j%2*w;y=j//2*(h+28);sheet.paste(im,(x,y));draw.text((x+12,y+h+5),f'{f/30:05.2f} s / frame {f}',fill='white')
 sheet.save(HERE/'contact-sheet.png')
 v1=MEDIA/'video_v1.mp4';assert sha(v1)=='d4f77f091a8d4084a52bc0f4761191f485f9feb6018da0948f22ae99f274be4e'
 report={'video_sha256':sha(movie),'video_bytes':movie.stat().st_size,'format':{'width':1920,'height':1080,'fps':30,'frames':960,'duration_seconds':float(probe['format']['duration']),'codec':'h264','audio':'aac','max_volume_dbfs':peak},'full_decode_errors':0,'shots':checks,'evidence_sha256':sha(HERE/'public/evidence.json'),'source_sha256':{n:sha(HERE/n) for n in ['src/Film.tsx','src/Root.tsx','export_evidence.py']},'representative_encoded_frames':[{'frame':f,'path':str(p.relative_to(HERE)),'sha256':sha(p)} for p,f in zip(images,frames)],'v1_preserved':True,'scope':'Verifies provenance, reveal timing and media integrity. Visual review of representative frames and Studio preview remains a separate editorial assessment.'}
 (HERE/'presentation-qa.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['video_bytes','format','full_decode_errors','v1_preserved']},indent=2))

if __name__=='__main__':main()
