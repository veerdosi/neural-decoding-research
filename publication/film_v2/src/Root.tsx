import {Composition, Folder} from 'remotion';
import {Film, Reconstruction} from './Film';
export const RemotionRoot = () => <>
  <Composition id="NeuralLanguageV2" component={Film} durationInFrames={960} fps={30} width={1920} height={1080}/>
  <Folder name="Reconstruction">
    <Composition id="Recording" component={Reconstruction} durationInFrames={180} fps={30} width={1920} height={1080} defaultProps={{index: 0, light: false, signalEnd: 1.6, wordStart: 2.05, wordStep: .23, truthStart: 3.45, fontSize: 174, accentWord: 'that', style: {backgroundColor: '#111611'}}}/>
  </Folder>
</>;
