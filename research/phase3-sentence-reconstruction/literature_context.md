# What would be new here?

This project is not the first demonstration of neural sentence decoding. The question is whether a frozen, general-purpose assistant can acquire a useful participant-specific decoding procedure during inference, and whether its interpretation or adaptation adds something beyond that procedure's numerical outputs.

[Willett et al., *A high-performance speech neuroprosthesis* (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468393/) used a trained recurrent neural decoder and language modeling for attempted-speech reconstruction. The paper also reports articulatory structure in speech representations. That supplies a biological reason to test transfer through phonetic features. It does not imply that a general-purpose language model can infer the neural mapping from a few dozen words.

[Kunz et al., *Inner speech in motor cortex and implications for speech neuroprostheses* (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360486/) is the source study for our sentence and sequence recordings. Its successful specialized decoders establish that these recordings can contain linguistic information. Reusing those data does not transfer the original paper's performance, online operation or inner-speech conclusions to our method. Our principal sentence results use cued attempted speech, with participant-specific calibration and offline processing.

[Nan et al., *Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding* (2026)](https://arxiv.org/abs/2604.08537) describes an explicitly meta-optimized model for in-context visual fMRI decoding in new subjects. “No fine-tuning” at evaluation therefore does not mean that a general-purpose model acquired the capability without specialized decoding training. It is relevant prior work against any broad claim that in-context brain decoding itself is new.

[Bosch et al., *Brain-language fusion enables interactive neural readout and in-silico experimentation* (2025)](https://arxiv.org/abs/2509.23941) trains CorText on visual fMRI to integrate brain activity into an LLM's latent space and support flexible language responses. Flexible neural-to-language interfaces also precede this project; the distinction under investigation is acquiring the task through supplied examples and general computation without such a separately trained neural interface.

These papers motivate a narrower contribution claim: an auditable investigation of inference-time neural decoding by an off-the-shelf general assistant, including attribution to numerical tools and tests of compositional transfer. The literature check is targeted, not an exhaustive priority search. A robust advantage over conventional decoders remains an empirical requirement, not an assumption.

Sources checked November19,2026. Preprints are identified by their linked arXiv records.
