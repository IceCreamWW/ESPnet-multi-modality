
#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh [--cmd (run.pl|queue.pl...)] <asr-exp-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  exit 1;
fi

asr_expdir=$1
name=$2
inference_tag=$3
for dir in ${asr_expdir}/${inference_tag}/${name}/score_wer; do
    [ -f local/score.py ] && python local/score.py --ref ${dir}/ref.trn --hyp ${dir}/hyp.trn
done
exit 0
