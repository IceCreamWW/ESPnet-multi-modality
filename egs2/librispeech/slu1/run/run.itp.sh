#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="valid test"

# asr_config=conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml
asr_config=conf/tuning/train_slu_transformer_2dec.yaml
inference_config=conf/decode_asr.yaml

dumpdir=/blob/tsst/users/v-weiwang1/espnet/egs2/librispeech/slu1/dump
expdir=/blob/tsst/users/v-weiwang1/espnet/egs2/librispeech/slu1/exp
datadir=/blob/tsst/users/v-weiwang1/espnet/egs2/librispeech/slu1/data
nltk_data_dir=/blob/tsst/users/v-weiwang1/nltk_data

./asr.itp.single.sh \
    --lang en \
    --ngpu 8 \
    --nbpe 10000 \
    --max_wav_duration 30 \
    --asr_tag "slu_tune_2dec" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir  "${dumpdir}" \
    --nltk_data_dir "${nltk_data_dir}" \
    --expdir "${expdir}" \
    --datadir "${datadir}" \
    --local_data_opts "--datadir ${datadir}" \
    --bpe_train_text "${datadir}/all_text" "$@"
