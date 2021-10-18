alignment_dir=/mnt/air-speech1/userdata/v-weiwang1/rawdata/librispeech/alignment
for dset in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
    dset_dir=$alignment_dir/$dset
    echo "processing $dset"
    python align.py --dir $dset_dir
    for f in phns phn_durs words word_durs; do
        cat $dset_dir/*/*/$f > $dset_dir/$f
    done
done
