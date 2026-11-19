Overview: The `seqRecallSpeech` task is an instructed delay task in which participants attempted to speak words associated with the directions of arrows to compare attempted speech behavior to the `seqRecallVerbalmemory` task (Figures S2 S3EF). 

Task design: In an instructed delay task, participants were presented with a text cue and attempted to speak those words. On each trial, the participant first saw a red square with text cue above it indicating what words to say (See "cueList" for each cue). Then the square turned green and the text cue was removed as a "go cue" after which the participant spoke the cued words. For T12, a sequence of direction words were cued that aligned with arrow sequences shown in the `seqRecallVerbalMemory` task (e.g. speech cue: "up right up", arrow sequence cue: ↑ → ↑). For T16 due to limited session time only single words were cued (e.g "up"). See "cueList" for the set of cues shown. Data was collected in a series of 'blocks', in between which the participant completed blocks of the `seqRecallVerbalMemory` task. Neural data between blocks is not recorded here. Data can be loaded with MATLAB or Python (scipy.io.loadmat).

Neural data files (file name indicates participant t12 or t16):
`t12.2023.09.28_binnedTX_spikePow_20ms_speak.mat`
`t16.2024.07.17_binnedTX_spikePow_20ms.mat`

Data fields:
spikePow: T x F matrix of binned spike band power, where T = number of time steps in the experiment and F = number of channels. Spike band power was defined as the mean of the squared voltages observed on the channel after high-pass filtering (250 Hz cutoff; units of microvolts squared). The data was denoised with a linear regression reference technique and block-wise mean subtracted to remove any nonstationairities which may be present in Utah array data. 

binnedTX: T x F matrix of binned threshold crossing counts, where T = number of time steps in the experiment and F = number of channels. The data was denoised with a linear regression reference technique and block-wise mean subtracted to remove any nonstationairities which may be present in Utah array data. Either a -3.5 or -4.5 x RMS threshold was used (see methods for individual participant threshold parameters). 

goTrialEpochs: N x 2 matrix of go period epochs (N = number of trials), where each row describes the starting and ending time step of the go period for that trial. (1-based indexing of 1st dimension of spikePow and binnedTX fields).

delayTrialEpochs: N x 2 matrix of delay period epochs (N = number of trials), where each row describes the starting and ending time step of the delay period for that trial. (1-based indexing of 1st dimension of spikePow and binnedTX fields).

trialCues: N x 1 vector of cue numbers (N = number of trials), where each entry describes the movement cue given on that trial. To get the text of that cue, use cueList (1-based indexing). For example to see the cue given on the 5th trial, use MATLAB code "cueList{trialCue(5)}" or Python code "cueList[0,trialCues[5]-1]".

cueList: C x 1 vector of cues (C = number of unique cues in the experiment), where each entry contains the text cue corresponding to that movement condition.

chanSets: A x 64 List of sets of channel indices organized by individual arrays (A = number of 64 channel Utah Arrays) (1-based indexing of 2nd dimension of spikePow and binnedTX fields).

chanSetNames: A x 1 List of string names of the array (A = number of 64 channel Utah arrays) corresponding to the sets of indices in chanSets. Names determined by the HCP cortical parcellation.

binSize: (int) Size in ms of the binned time steps, T in `spikePow` and `binnedTX` neural features. Either 20 or 10 depending on the participant. 

