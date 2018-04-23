#!/bin/bash
# Copyright   2017 Yiwen Shao

nj=4
cmd=run.pl
scale_size=40
echo "$0 $@"

. utils/parse_options.sh || exit 1;

data=$1
featdir=$data/data
scp=$data/images.scp
logdir=data/local/log

# make $featdir an absolute pathname
featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

for n in $(seq $nj); do
    split_scps="$split_scps $logdir/images.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;


$cmd JOB=1:$nj $logdir/extract_feature.JOB.log \
  local/make_features.py $logdir --feat-dim 40 --job JOB \| \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:$featdir/images.JOB.ark,$featdir/images.JOB.scp

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featdir/images.$n.scp || exit 1;
done > $data/feats.scp || exit 1

