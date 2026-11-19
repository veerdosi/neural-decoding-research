Overview: The interleaved verbal behaviors task refers to blocks in which the participant performed the same 7 word and do nothing cue set as in the isolated verbal behaviors task, however within each block an attempted, an imagined and listening cue types were intermixed. 
These data (contained here in this folder) can be used to assess neural tuning relationships between attempted, imagined and perceived speech behaviors.  These were instructed delay experiments with text-based cues (or audio cues for listening). This data is depicted in Figures 5.

spikePow: T x F matrix of binned spike band power, where T = number of time steps in the experiment and F = number of channels. Spike band power was defined as the mean of the squared voltages observed on the channel after high-pass filtering (250 Hz cutoff; units of microvolts squared). The data was denoised with a linear regression reference technique. 

binnedTX: T x F matrix of binned threshold crossing counts, where T = number of time steps in the experiment and F = number of channels. The data was denoised with a linear regression reference technique and either a -3.5 or -4.5 x RMS threshold was used (see methods for individual participant threshold parameters). 

blockNum: T x 1 vector of block numbers (T = number of time steps), which describe for each time step what block number that time step was recorded during. Note that gaps in the recordings occur when the block number changes (neural activity in between blocks was not recorded).

goTrialEpochs: N x 2 matrix of go period epochs (N = number of trials), where each row describes the starting and ending time step of the go period for that trial.

delayTrialEpochs: N x 2 matrix of delay period epochs (N = number of trials), where each row describes the starting and ending time step of the delay period for that trial.

trialCues: N x 1 vector of cue numbers (N = number of trials), where each entry describes the movement cue given on that trial. To get the text of that cue, use cueList. For example to see the cue given on the 5th trial, use MATLAB code "cueList{trialCue(5)}" or Python code "cueList[0,trialCues[5]-1]".

cueList: C x 1 vector of cues (C = number of unique cues in the experiment), where each entry contains the text cue corresponding to that movement condition.

chanSets: List of sets of channel indices making up individual arrays.

chanSetNames: List of string names of the array names (as determined by the HCP cortical parcellation) corresponding to the sets of indices in chanSets. 

binSize: (int) Size in ms of the binned time steps, T in above variables. Either 20 or 10 depending on the participant. 

