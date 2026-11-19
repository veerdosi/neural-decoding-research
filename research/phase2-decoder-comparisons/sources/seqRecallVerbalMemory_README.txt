Overview: The `seqRecallVerbalMemory` task probes the representation of instructed inner speech for sequence recall. When compared to the `seqRecallVisualMemory` task, it probes the volitional nature of inner speech for cognitive tasks. When compared to `seqRecallSpeech` it probes the relationship between inner speech for sequence recall and attempted speaking. (Figure 4CD, Figure S3C-F)

Task design: In an instructed delay task, participants were presented with a cue consisting of a sequence of arrows which indicated a sequence of hand movements. On each trial, the participant first saw a red square with the arrow cue above it indicating the instructed movement sequence. Then the square turned green and the cue was removed after which the participant attempted to draw lines sequentially in the direction of the arrows (e.g. arrow sequence ↑ → ↑ cues drawing three connected line segments, first moving up, then right, then up). Participants were explicitly cued to use an inner speech based mental strategy to remember the and reproduce the sequence. Both participants used their dominant right arm. See "cueList" for the set of cues shown. Data was collected in a series of 'blocks', in between which the participant completed blocks of the `seqRecallVisualMemory` task and for T16 only also the `seqRecallSpeech` task. Neural data between blocks is not recorded here. Data can be loaded with MATLAB or Python (scipy.io.loadmat).

Neural data files (file name indicates participant t12 or t16):
`t12.2023.09.28_binnedTX_spikePow_20ms_arrow.mat`
`t12.2023.10.12_binnedTX_spikePow_20ms_verbalMemory.mat`
`t16.2024.07.17_binnedTX_spikePow_20ms_verbal.mat`

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