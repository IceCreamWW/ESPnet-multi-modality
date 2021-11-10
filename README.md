​	

# Optimizing Alignment of Speech and Language Latent Spaces for End-to-End Speech Recognition and Understanding

## Directories

### raw data

ASR: 

- /tsstd01scus/users/v-weiwang1/rawdata/librispeech/
-  /tsstd01wus2/users/v-weiwang1/rawdata/librispeech/

SLU: 

- /tsstd01scus/users/v-weiwang1/rawdata/SNIPS/

### dump data

ASR: 

- /tsstd01scus/users/v-weiwang1/espnet/egs2/librispeech/asr2/dump/

SLU: 

- /tsstd01scus/users/v-weiwang1/espnet2/egs2/librispeech/asr2/slu/dump/

### Experiments

ASR: 

- /tsstd01scus/users/v-weiwang1/espnet/egs2/librispeech/asr2/exp/

SLU: 

- /tsstd01scus/users/v-weiwang1/espnet2/egs2/librispeech/asr2/exp/

## Train

### stage 0: data preparation

paired data is prepared with espnet2 recipe, no custom scripts are required. 

unpaired data is the filtered [gutenburg corpus](https://www.openslr.org/resources/12/original-books.tar.gz), three types of unpaired text are dropped:

- with repeatitive characters at start
- g2p result longer than the longest text in paired transcripts
- g2p result shorter than or equal to the shortest text in paired transcripts 

### stage 1: dump data

dump is performed with standard espnet recipe, data is dump directory should be ready for training.

### stage 2: training

#### ASR

Run `amlt run philly/run.s12t6share2.mse_ctc01.yaml` 

#### SLU
Run `amlt run philly/run.slu.yaml`

##### parameter explanation (in config file)

`use_mse_ctc`: use Euclidean or dot product as distance metrices

`use_bert_Ctc`: use MLM loss or not (if set to`True`, the ctc type will always be same as `use_mse_ctc`)

set both `mse_ctc_weight` to and `bert_ctc_Weight` to 0 can disable the loss at encoder end (set `use_mse_ctc` or `use_bert_ctc`  to `False` won't make sense since it controls the ctc type)

##### tips for reproduction

if the project is reproduced on other tools, some details should be taken care of:

- The online torch stft by default use 512 windows length and 128 hop lengths. However, to match the force alignment result, windows length should be set to 400 and hop length to 160.
- If g2p_en is used to perform g2p operation, the default cmudict should be replaced with official librispeech lexicon. utterances that have different phonemes in force alignment result and g2p result should not perform swap embedding.
- For rep-phoneme, silence at begin and end should be modelled differently as silence in middle of a sentence.
- swap embedding cannot be done by `a,b = b,a` since tensors with grad use shallow copy by default, check the mask implementation in this repo.
- swap embedding should be added after the model is close to convergence

## Inference

Since multi-processing decoding meets some unknown issue and some of decoding processes might be killed. Run ```amlt run philly/run.decode.yaml``` for a `sleep` job, ssh to the target machine and start the decoding manually.


