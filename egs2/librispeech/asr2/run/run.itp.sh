#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

dumpdir=/blob/tsst/users/v-weiwang1/librispeech/asr2/dump
expdir=/blob/tsst/users/v-weiwang1/librispeech/asr2/exp
datadir=/blob/tsst/users/v-weiwang1/librispeech/asr2/exp
nltk_data_dir=/blob/tsst/users/v-weiwang1/nltk_data

./asr.itp.sh \
    --lang en \
    --ngpu 8 \
    --nbpe 10000 \
    --g2p g2p_en \
    --max_wav_duration 30 \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir  "${dumpdir}" \
    --nltk_data_dir "${nltk_data_dir}" \
    --expdir "${expdir}" \
    --datadir "${datadir}" \
    --local_data_opts "--datadir ${datadir}" \
    --bpe_train_text "${datadir}/local/other_text/text" "$@"
