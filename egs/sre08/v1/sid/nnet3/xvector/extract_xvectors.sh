#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=30
cmd="run.pl"
chunk_size=-1 # The chunk size over which the embedding is extracted.
              # If left unspecified, it uses the max_chunk_size in the nnet
              # directory.
use_gpu=false
stage=0
apply_exp=false
extract_config=
min_frames=25
collapse_model=true
batchnorm_test_mode=true
raw=false
per_utt=false
model=final
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/${model}.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat $srcdir/min_chunk_size 2>/dev/null`
max_chunk_size=`cat $srcdir/max_chunk_size 2>/dev/null`
nnet=$srcdir/${model}.raw
if [ -f $srcdir/$extract_config ] ; then
  echo "$0: using $srcdir/$extract_config to extract xvectors"
  nnet="nnet3-copy --nnet-config=$srcdir/$extract_config $srcdir/${model}.raw - |"
fi

# generate subset of data with utt lenght larger than $num_frames to be computable by model.
name=`basename $data`
tmp=$srcdir/${name}_tmp
mkdir -p $tmp
if [ ! -f $data/utt2num_frames ]; then
  feat-to-len scp:$data/feats.scp ark,t:$data/utt2num_frames
fi
cat $data/utt2num_frames  | awk -v minf=$min_frames '{if ($2 > minf) print $1}'  > $tmp/utt_list
./utils/subset_data_dir.sh --utt-list $tmp/utt_list $data $tmp/data
data=$tmp/data

if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi

mkdir -p $dir/log

per_utt_opt=
per_utt_suff=
if ${per_utt};then
  per_utt_opt="--per-utt"
  per_utt_suff="utt"
fi

utils/split_data.sh ${per_utt_opt} $data $nj
echo "$0: extracting xvectors for $data"
sdata=$data/split${nj}${per_utt_suff}/JOB

# Set up the features
if $raw; then
  feat="ark:select-voiced-frames scp:${sdata}/feats.scp scp,s,cs:${sdata}/vad.scp ark:- |"
else
  feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"
fi

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd --gpu 1 ${dir}/log/extract.$g.log \
        nnet3-xvector-compute --batchnorm-test-mode=$batchnorm_test_mode --collapse-model=$collapse_model \
          --use-gpu=yes --apply-exp=$apply_exp --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size \
          "$nnet" "`echo $feat | sed s/JOB/$g/g`" ark,scp:${dir}/xvector.$g.ark,${dir}/xvector.$g.scp || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet3-xvector-compute --batchnorm-test-mode=$batchnorm_test_mode --collapse-model=$collapse_model \
        --use-gpu=no --apply-exp=$apply_exp --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size \
        "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "$0: computing mean of xvectors for each speaker"
  $cmd $dir/log/speaker_mean.log \
    ivector-mean ark:$data/spk2utt scp:$dir/xvector.scp \
    ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
fi
