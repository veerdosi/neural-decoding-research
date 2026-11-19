Overview: These were instructed delay sentence-production experiments. On each trial, the participant first saw a red square with a sentence above it (see the "sentenceText" variable). Then, when the square turned green, the participant either attempted to speak that sentence normally or without vocalization, or imagined speaking the sentence with their preferred inner speech strategy. When finished, the participant either pushed a button (T12) or indicated to the researcher to push the button by looking at them (T16), used an eye tracker to push a button on the screen (T15). Data was collected in a series of 'blocks' (20-50 sentences in each block), in between which the participant rested. Neural data between blocks was not recorded. During some of the blocks, the sentences decoded in real-time and the output of the decoder was displayed on the screen. Some sets of sentences were from a small vocabulary of 50 words, and some were from an open vocabulary of approximately 130,000 words. Files are named to indicate participant session, behavior performed, vocabulary size and whether it was open or closed-loop (indicating sentences were being decoded in real-time). 

spikePow: T x F matrix of binned spike band power, where T = number of time steps in the experiment and F = number of channels. Spike band power was defined as the mean of the squared voltages observed on the channel after high-pass filtering (250 Hz cutoff; units of microvolts squared). The data was denoised with a linear regression reference technique.

binnedTX: T x F matrix of binned threshold crossing count neural features, where T = number of time steps in the experiment and F = number of channels. The data was denoised with a linear regression reference technique. 

blockNum : T x 1 vector of block numbers (T = number of time steps), which describe for each time step what block number that time step was recorded during.

goTrialEpochs: N x 2 matrix of go period epochs (N = number of trials), where each row describes the starting and ending time step of the go period for that trial.

delayTrialEpochs: N x 2 matrix of delay period epochs (N = number of trials), where each row describes the starting and ending time step of the go period for that trial.

sentenceText: N x 1 vector of sentences (N = number of trials), where each entry contains the sentence that was displayed on the screen for that trial.

sentenceFeatures: N x 1 list containing neural features corresponding to each sentence (N = number of trials). The data begins at the go cue (no delay period activity is included). The first half of the features are the "binnedTX" features and the last half are the "spikePow" features. The "sentenceFeatures" variable was included for convenience and can be reconstructed from other time series variable in this file. 

sentenceDurations: N x 1 vector of sentence durations (N = number of trials), where each entry describes the duration of the go period (sentence production period) corresponding to that trial. Durations are in number of time bins. 

binSize: integer indicating the size in ms of the time bins (10 or 20 depending on the participant).

decodedSentenceText: N x 1 vector of final decoded sentence for closed-loop blocks. (N = number of trials), where each entry contains the final decoded sentence from the rnn and language model. Open-loop blocks have empty strings. 

chanSets: List of sets of channel indices making up individual arrays.

chanSetNames: List of string names of the array names (as determined by the HCP cortical parcellation) corresponding to the sets of indices in chanSets. 