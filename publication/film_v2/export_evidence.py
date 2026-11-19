"""Export four verified reconstructions and their own neural recordings for v2."""
import hashlib,json,pathlib,shutil,wave
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from PIL import Image

HERE=pathlib.Path(__file__).resolve().parent;MEDIA=HERE.parent;PROJECT=MEDIA.parent;ROOT=PROJECT/'research/phase3-sentence-reconstruction'
import sys
sys.path.insert(0,str(ROOT/'code'))
from paths import resolve_sandbox
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 selections=[('sentences_t16_confirm','sentences_t16_confirm_repeat','tcd92c113',6.0,[['where','is','that']]),('sentences_t16_confirm','sentences_t16_confirm_repeat','tef9d94a6',7.0,[['do','they','have'],['my','glasses']]),('sentences_t16_confirm','sentences_t16_confirm_repeat','tf75891ff',8.0,[['my','family','is'],['not','very','tired']]),('sentences_t12_fresh','sentences_t12_fresh_repeat','tdd25032b',11.0,[['tell','my','family'],['i','do','not','feel','thirsty']])]
 shots=[];sources={};offset=0
 for index,(name,repeat,qid,duration,lines) in enumerate(selections):
  case=read(ROOT/'cases'/f'{name}.json');rows=read(ROOT/'evaluation'/f'{name}.sol.json')['rows'];r=next(r for r in rows if r['id']==qid);rr=next(r for r in read(ROOT/'evaluation'/f'{repeat}.sol.json')['rows'] if r['id']==qid);assert r['prediction']==r['truth']==rr['prediction']==rr['truth'];assert [w for line in lines for w in line]==r['prediction']
  folder=resolve_sandbox(case['sandbox']);meta=read(folder/'inputs.json');assert not any(' '+' '.join(r['truth'])+' ' in ' '+' '.join(c['tokens'])+' ' for c in meta['calibration'])
  source=PROJECT/'local/sources'/case['source'];assert sha(source)==case['source_sha256']
  if str(source) not in sources:sources[str(source)]=loadmat(source,variable_names=['binnedTX','goTrialEpochs','binSize'],simplify_cells=True)
  d=sources[str(source)];start,end=d['goTrialEpochs'][r['trial']].astype(int)-1;channels=case['channel_indices'][:64];tx=np.asarray(d['binnedTX'][start:end+1][:,channels],float);seconds=len(tx)*float(d['binSize'])/1000;native=tx.astype(int)
  # One pixel per original channel/bin. Alpha maps counts; no invented events.
  alpha=np.round(np.clip(tx.T/4,0,1)*255).astype('uint8')
  for theme,color in [('dark',(238,234,225)),('light',(20,23,21))]:
   image=np.zeros((*alpha.shape,4),dtype='uint8');image[:,:,:3]=color;image[:,:,3]=alpha;Image.fromarray(image).save(HERE/'public'/f'recording-{index+1}-{theme}.png')
  population=gaussian_filter1d(tx.mean(1),2);lo,hi=np.quantile(population,[.01,.99]);trace=np.clip((population-lo)/max(hi-lo,1e-9),0,1)
  signals={'counts':native.tolist(),'bin_ms':float(d['binSize']),'display_channels_zero_based':channels,'population_trace_normalized':np.round(trace,5).tolist(),'display_alpha':'Native-bin counts/4, clipped0..1. Each source bin/channel retained. No fabricated spike times.','population_trace':'Mean crossing counts over displayed64 channels, Gaussian sigma2 native bins, 1st/99th percentile display scaling only.'}
  signal_path=HERE/'public'/f'recording-{index+1}.json';signal_path.write_text(json.dumps(signals,separators=(',',':'))+'\n')
  shots.append({'index':index+1,'source_case':name,'repeat_case':repeat,'query_id':qid,'participant':case['source'].split('.')[0].upper(),'source_file':case['source'],'source_sha256':case['source_sha256'],'trial_zero_based':r['trial'],'calibration_n':case['calibration_n'],'prediction':r['prediction'],'target':r['truth'],'lines':lines,'source_duration_seconds':seconds,'duration_seconds':duration,'from_seconds':offset,'signal_seconds':[1.6,1.8,2.1,2.4][index],'word_start_seconds':[2.05,2.25,2.65,3.05][index],'word_step_seconds':[.23,.23,.24,.27][index],'truth_start_seconds':[3.45,4.35,5.0,6.25][index],'signal_asset':signal_path.name,'signal_sha256':sha(signal_path),'dark_raster':f'recording-{index+1}-dark.png','light_raster':f'recording-{index+1}-light.png','trace':np.round(trace[::2],5).tolist(),'evidence_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'responses'/f'{name}.json',ROOT/'responses'/f'{repeat}.json',ROOT/'evaluation'/f'{name}.sol.json',ROOT/'cases'/f'{name}.json']}});offset+=duration
 evidence={'version':'v2','model':'gpt-6.1-sol','reasoning_effort':'high','duration_seconds':offset,'shots':shots,'selection':'Four actual held-out strict-novel full targets, all exact in two independent calls. Manually selected for ascending length3/5/6/8 words and meaningful sentences. Selected successes, not a representative accuracy sample.','scientific_scope':'Cued attempted speech from prerecorded intracortical recordings. Calibration-specific frozen Sol with general numerical tools. Signal displays are excerpts of the neural inputs, not word-aligned biological events.','presentation':'Completed inference replay. Edited timing and word reveals are presentation choices, not inference latency or measured neural word boundaries.'}
 (HERE/'public/evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 # Original quiet mechanical sound design. Artistic audio, not brain/voice audio.
 rate=48000;rng=np.random.default_rng(6081);peaks={}
 for name,freq,amp,length in [('pulse',95,.13,.18),('word',390,.10,.09),('confirm',520,.15,.24)]:
  z=np.arange(round(length*rate))/rate;envelope=(1-np.exp(-z*700))*np.exp(-z*(40 if name!='confirm' else 24));sound=amp*envelope*(np.sin(2*np.pi*freq*z)+.12*rng.normal(size=len(z)))
  if name=='confirm':sound+=.045*np.exp(-np.maximum(z-.07,0)*38)*(z>.07)*np.sin(2*np.pi*780*z)
  sound=np.tanh(sound);pcm=(np.stack([sound,sound],axis=1)*32767).astype('<i2')
  with wave.open(str(HERE/'public'/f'{name}.wav'),'wb') as out:out.setnchannels(2);out.setsampwidth(2);out.setframerate(rate);out.writeframes(pcm.tobytes())
  peaks[name]=float(20*np.log10(max(abs(sound))))
 (HERE/'public/instrument.wav').unlink(missing_ok=True)
 (HERE/'public/sound-design.json').write_text(json.dumps({'source':'Original deterministic synthesis, artistic mechanical and confirmation clicks. No participant voice or neural audio.','seed':6081,'sample_rate':rate,'sample_peak_dbfs':peaks,'timing':'Audio follows source-editable scene/word/reveal timings in Remotion.'},indent=2)+'\n')
 print('Exported',len(shots),'verified recordings;',offset,'seconds')

if __name__=='__main__':main()
