import {Audio} from '@remotion/media';
import {loadFont} from '@remotion/fonts';
import {CanvasImage, Interactive, Track, continueRender, delayRender, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import data from '../public/evidence.json';

const fontHandle = delayRender('Bundled IBM Plex fonts');
Promise.all([
  loadFont({family: 'IBM Plex Sans', url: staticFile('ibm-plex-sans-latin-400-normal.woff2'), weight: '400'}),
  loadFont({family: 'IBM Plex Sans', url: staticFile('ibm-plex-sans-latin-500-normal.woff2'), weight: '500'}),
  loadFont({family: 'IBM Plex Sans', url: staticFile('ibm-plex-sans-latin-600-normal.woff2'), weight: '600'}),
  loadFont({family: 'IBM Plex Mono', url: staticFile('ibm-plex-mono-latin-400-normal.woff2'), weight: '400'}),
]).then(() => continueRender(fontHandle));
const clamp = {extrapolateLeft: 'clamp' as const, extrapolateRight: 'clamp' as const};
const mono = 'IBM Plex Mono';
const sans = 'IBM Plex Sans';
const orange = '#d2643d';
const palette = (light: boolean) => light ? {ink: '#161b18', grey: '#666b64', rule: '#c9c8bf'} : {ink: '#eeeae1', grey: '#989d94', rule: '#353c35'};
type Shot = typeof data.shots[number];
type SceneProps = {index: number; light: boolean; signalEnd: number; wordStart: number; wordStep: number; truthStart: number; fontSize: number; accentWord: string};

const Header = ({index, light, signalEnd}: Pick<SceneProps, 'index' | 'light' | 'signalEnd'>) => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig(); const c = palette(light);
  return <>
    <Interactive.Div name="Brain activity label" style={{position: 'absolute', left: 144, top: 116, color: c.ink, fontFamily: sans, fontSize: 58, fontWeight: 400, letterSpacing: -1.5, opacity: interpolate(f, [signalEnd * fps, (signalEnd + .22) * fps], [1, 0], clamp)}}>Brain activity</Interactive.Div>
    <Interactive.Div name="Reconstruction label" style={{position: 'absolute', left: 144, top: 116, color: c.ink, fontFamily: sans, fontSize: 58, fontWeight: 400, letterSpacing: -1.5, opacity: interpolate(f, [(signalEnd + .12) * fps, (signalEnd + .38) * fps], [0, 1], clamp)}}>Sol's reconstruction</Interactive.Div>
    <Interactive.Div name="Model identity" style={{position: 'absolute', right: 330, top: 137, color: c.ink, fontFamily: mono, fontSize: 32}}>GPT-6.1 Sol</Interactive.Div>
    <div style={{position: 'absolute', right: 144, top: 137, color: orange, fontFamily: mono, fontSize: 32}}>{String(index + 1).padStart(2, '0')} / 04</div>
    <div style={{position: 'absolute', left: 144, right: 144, top: 212, height: 1, backgroundColor: c.rule}}/>
  </>;
};

const Signal = ({shot, light, signalEnd}: {shot: Shot; light: boolean; signalEnd: number}) => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig(); const c = palette(light);
  const top = interpolate(f, [signalEnd * fps, (signalEnd + .48) * fps], [310, 855], clamp);
  const height = interpolate(f, [signalEnd * fps, (signalEnd + .48) * fps], [430, 54], clamp);
  const traceHeight = interpolate(f, [signalEnd * fps, (signalEnd + .48) * fps], [42, 16], clamp);
  const progress = Math.min(1, f / (signalEnd * fps));
  const path = shot.trace.map((v, i) => `${i ? 'L' : 'M'}${(i / (shot.trace.length - 1) * 1632).toFixed(2)},${((1-v)*traceHeight).toFixed(2)}`).join(' ');
  const raster = staticFile(light ? shot.light_raster : shot.dark_raster);
  return <div style={{opacity: interpolate(f, [signalEnd * fps, (signalEnd + .6) * fps], [1, .44], clamp)}}>
    <div style={{position: 'absolute', left: 144, top, width: 1632, height}}>
      {[0, 16, 32, 48, 64].map(row => <div key={row} style={{position: 'absolute', left: 0, right: 0, top: row/64*height, height: 1, backgroundColor: c.rule, opacity: .5}}/>)}
      <CanvasImage src={raster} style={{position: 'absolute', width: '100%', height: '100%', imageRendering: 'pixelated', opacity: .19}}/>
      <div style={{position: 'absolute', inset: 0, clipPath: `inset(0 ${(1-progress)*100}% 0 0)`}}><CanvasImage src={raster} style={{width: '100%', height: '100%', imageRendering: 'pixelated'}}/></div>
      <div style={{position: 'absolute', left: progress*1632, top: -12, height: height + 24, width: 2, backgroundColor: orange, opacity: interpolate(f, [signalEnd*fps, (signalEnd+.3)*fps], [1, 0], clamp)}}/>
    </div>
    <svg width={1632} height={traceHeight + 2} style={{position: 'absolute', left: 144, top: top+height+14, overflow: 'visible'}}><path d={path} fill="none" stroke={c.ink} strokeWidth={1.5}/></svg>
    <div style={{position: 'absolute', left: 144, top: top+height+traceHeight+27, fontFamily: mono, fontSize: 24, color: c.grey}}>0.00 s</div>
    <div style={{position: 'absolute', right: 144, top: top+height+traceHeight+27, fontFamily: mono, fontSize: 24, color: c.grey}}>{shot.source_duration_seconds.toFixed(2)} s</div>
  </div>;
};

const Words = ({shot, light, wordStart, wordStep, fontSize, accentWord}: {shot: Shot} & Pick<SceneProps, 'light' | 'wordStart' | 'wordStep' | 'fontSize' | 'accentWord'>) => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig(); const c = palette(light);
  return <div style={{position: 'absolute', left: 144, right: 144, top: shot.lines.length === 1 ? 350 : 292, fontFamily: sans, fontSize, fontWeight: 500, letterSpacing: -4.5, lineHeight: 1.08}}>
    {shot.lines.map((line, row) => <div key={row} style={{whiteSpace: 'nowrap'}}>{line.map((word, j) => {
      const index = shot.lines.slice(0, row).flat().length+j;
      return <span key={j} data-role="decoded-word" style={{display: 'inline-block', marginRight: fontSize * .24, color: word === accentWord ? orange : c.ink, opacity: interpolate(f, [(wordStart+index*wordStep)*fps, (wordStart+index*wordStep+.17)*fps], [0, 1], clamp), translate: `0px ${interpolate(f, [(wordStart+index*wordStep)*fps, (wordStart+index*wordStep+.22)*fps], [16, 0], clamp)}px`}}>{word}</span>;
    })}</div>)}
  </div>;
};

const Target = ({shot, light, truthStart}: {shot: Shot} & Pick<SceneProps, 'light' | 'truthStart'>) => {
  const f = useCurrentFrame(); const {fps} = useVideoConfig(); const c = palette(light);
  return <>
    <div style={{position: 'absolute', left: 144, right: 144, top: 662, height: 1, backgroundColor: c.rule, clipPath: `inset(0 ${interpolate(f, [truthStart*fps, (truthStart+.3)*fps], [100, 0], clamp)}% 0 0)`}}/>
    <Interactive.Div name="Ground truth label" style={{position: 'absolute', left: 144, top: 704, fontFamily: mono, fontSize: 29, letterSpacing: .5, color: orange, opacity: interpolate(f, [truthStart*fps, (truthStart+.2)*fps], [0, 1], clamp)}}>TARGET</Interactive.Div>
    <div data-role="target-sentence" style={{position: 'absolute', left: 350, top: 689, right: 144, fontFamily: sans, fontSize: 52, fontWeight: 400, letterSpacing: -.8, color: c.ink, clipPath: `inset(0 ${interpolate(f, [(truthStart+.12)*fps, (truthStart+.48)*fps], [100, 0], clamp)}% 0 0)`}}>{shot.target.join(' ')}</div>
  </>;
};

const Footer = ({light}: {light: boolean}) => {
  const c = palette(light);
  return <>
    <Interactive.Div name="Recording context" style={{position: 'absolute', left: 144, top: 1000, fontFamily: mono, fontSize: 27, color: c.grey}}>Recorded attempted speech</Interactive.Div>
    <Interactive.Div name="Replay context" style={{position: 'absolute', right: 144, top: 1000, fontFamily: mono, fontSize: 27, color: c.grey}}>Selected results · inference replay</Interactive.Div>
  </>;
};

const ReconstructionInner = ({index, light, signalEnd, wordStart, wordStep, truthStart, fontSize, accentWord}: SceneProps) => {
  const {fps} = useVideoConfig(); const shot = data.shots[index];
  return <>
    <Header index={index} light={light} signalEnd={signalEnd}/>
    <Signal shot={shot} light={light} signalEnd={signalEnd}/>
    <Words shot={shot} light={light} wordStart={wordStart} wordStep={wordStep} fontSize={fontSize} accentWord={accentWord}/>
    <Target shot={shot} light={light} truthStart={truthStart}/>
    <Footer light={light}/>
    <Track name="Instrument pulse"><Audio name="Recording cut" src={staticFile('pulse.wav')} durationInFrames={Math.round(.18*fps)} volume={.9}/></Track>
    <Track name="Word reveals">{shot.prediction.map((_, i) => <Audio key={i} name={`Word ${i+1}`} src={staticFile('word.wav')} from={Math.round((wordStart+i*wordStep)*fps)} durationInFrames={Math.round(.1*fps)} volume={.8}/>)}</Track>
    <Track name="Truth reveal"><Audio name="Confirmation" src={staticFile('confirm.wav')} from={Math.round(truthStart*fps)} durationInFrames={Math.round(.25*fps)} volume={.9}/></Track>
  </>;
};
export const Reconstruction = Interactive.create({Component: ReconstructionInner, layout: 'absolute-fill', schema: {
  ...Interactive.backgroundSchema,
  index: {type: 'number', hiddenFromList: false, default: 0, min: 0, max: 3, integer: true, description: 'Verified recording index'},
  light: {type: 'boolean', default: false, description: 'Warm white final scene'},
  signalEnd: {type: 'number', hiddenFromList: false, default: 1.6, min: .1, description: 'Recording scan duration (seconds)'},
  wordStart: {type: 'number', hiddenFromList: false, default: 2.05, min: 0, description: 'Decoded-word reveal starts (seconds)'},
  wordStep: {type: 'number', hiddenFromList: false, default: .23, min: .01, description: 'Spacing between word reveals (seconds)'},
  truthStart: {type: 'number', hiddenFromList: false, default: 3.45, min: 0, description: 'Ground-truth reveal starts (seconds)'},
  fontSize: {type: 'number', hiddenFromList: false, default: 174, min: 60, description: 'Decoded type size'},
  accentWord: {type: 'string', default: 'that', description: 'Editorial orange word highlight'},
}});

export const Film = () => {
  const {fps} = useVideoConfig();
  return <Track name="Four reconstructions">
    <Reconstruction name="01 — where is that" from={0} durationInFrames={6*fps} index={0} light={false} signalEnd={1.6} wordStart={2.05} wordStep={.23} truthStart={3.45} fontSize={174} accentWord="that" style={{backgroundColor: '#111611'}}/>
    <Reconstruction name="02 — my glasses" from={6*fps} durationInFrames={7*fps} index={1} light={false} signalEnd={1.8} wordStart={2.25} wordStep={.23} truthStart={4.35} fontSize={144} accentWord="glasses" style={{backgroundColor: '#111611'}}/>
    <Reconstruction name="03 — my family" from={13*fps} durationInFrames={8*fps} index={2} light={false} signalEnd={2.1} wordStart={2.65} wordStep={.24} truthStart={5} fontSize={144} accentWord="not" style={{backgroundColor: '#111611'}}/>
    <Reconstruction name="04 — tell my family" from={21*fps} durationInFrames={11*fps} index={3} light={true} signalEnd={2.4} wordStart={3.05} wordStep={.27} truthStart={6.25} fontSize={140} accentWord="family" style={{backgroundColor: '#eeeae1'}}/>
  </Track>;
};
