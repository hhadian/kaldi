#!/bin/bash

# Copyright 2017 Pegah Ghahremani
# Apache 2.0

# This script evaluates age detection model trained using nnet3 model.

[ -f ./path.sh ] && . ./path.sh; # source the path.

utt2label_file=utt2age # filename in data dir, that consists (utt, label) pairs
train_post=      # training log-posterior dir, if provided, xvector.scp in
                 # this dir is used to computed
                 # the log-prior and it is subtracted from test posterior to have uniform prior.
                 # e.g. data/train_30k/xvector.scp
prior_scale=-0.85
use_test_prior=false # If provided, the xvector.scp in this directory is used
                     # to compute training log prior and
                     # the model is reballanced to have prior equivalent to test prior.
age2class=           # the mapping from original class to new classes used to train model.
                     # This file can be generated using local/prepare_age_to_class_map.py
                     # If empty, it assumes the file is in xvector data dir.
per_frame_output=false # If true, the output is per-frame-output (lstm network), otherwise
                       # single frame is per utt for output.
log_posterior=true   # If true, it assumes train_post is in log domain.
. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <data-dir> <xvector-data>";
  echo "e.g.: $0 data/sre_08_10/1/test exp/xvector_age/1/test"
fi
data=$1
xvector_data=$2

dir=`dirname $data`
model_dir=`dirname $xvector_data`
apply_exp=false
if $log_posterior; then apply_exp=true ;fi

if [ -z $age2class ]; then
  #age2class=$xvector_data/age2class
  age2class=$model_dir/age2class
  echo "$0: $age2class is used as age2class."
fi

if [ ! -f $age2class ]; then
  echo "$0: There is no age2class file $age2class."
  exit 1;
fi
if $per_frame_output;then
  echo "$0: average posterior probability for all frames in utterances."
  train_xvec="ark:copy-matrix --apply-exp=$apply_exp scp:$train_post/xvector.scp ark:- | matrix-sum-rows --average=true ark:- ark:- |"
  test_xvec="ark:copy-matrix --apply-exp=$apply_exp scp:$xvector_data/xvector.scp ark:- | matrix-sum-rows --average=true ark:- ark:- |"
  apply_exp=false # it is already applied.
else
  train_xvec="scp:$train_post/xvector.scp"
  test_xvec="scp:$xvector_data/xvector.scp"
fi

if [ -f $train_post/xvector.scp ]; then
  num_test_utt=`wc -l $data/$utt2label_file | awk '{print $1}'`

  echo "$0: create uniform log prior to rebalance the model."
  copy-vector --apply-exp=$apply_exp "$train_xvec" ark:- | \
  vector-sum --binary=false --average=true ark:- $xvector_data/prior.from.post.vec

  num_pdf=`wc -w $xvector_data/prior.from.post.vec | awk '{print $1-2}'`
  utt2label_file_tmp=$data/${utt2label_file}_back_off
  rm -rf $utt2label_file_tmp
  cp $data/$utt2label_file $utt2label_file_tmp
  for n in `cat $age2class | awk '{print $1}'`;do
    echo "utt_tmp_$n $n" >> $utt2label_file_tmp
  done
  test_prior=""
  if $use_test_prior; then
    echo "$0: generate log-priors for test data."
    utils/apply_map.pl -f 2 $age2class < ${utt2label_file_tmp} | awk '{print $2}' | sort -n | uniq -c | \
    awk -v p=$prior_scale -v s=$[$num_test_utt+$num_pdf] 'BEGIN{printf(" [ ");} {printf("%s ", -p*log(($1)/s)); } END{print ("]"); }' > $xvector_data/prior.test.vec
    for n in $(seq 1 $num_test_utt);do cat $xvector_data/prior.test.vec; done > $xvector_data/prior.test.tmp
    paste <(awk '{print $1}' $data/${utt2label_file}) <(cat $xvector_data/prior.test.tmp) > $xvector_data/utt2log.prior.test
    test_prior=ark:$xvector_data/utt2log.prior.test
  fi

  for n in $(seq 1 $num_test_utt);do cat $xvector_data/prior.from.post.vec; done > $xvector_data/prior.from.post.tmp
  paste <(awk '{print $1}' $data/${utt2label_file}) <(cat $xvector_data/prior.from.post.tmp) > $xvector_data/utt2prior.train


  # sum -log-prior from posterior to reballance the model to have uniform prior.
  copy-vector --apply-log=true ark:${xvector_data}/utt2prior.train ark:- | \
    copy-vector --scale=$prior_scale ark:- ark:- | \
    vector-sum ark:- $test_prior "$test_xvec" ark,t:$xvector_data/norm_post
fi

if [ -f $xvector_data/xvector.scp ]; then
  for x in $xvector_data/spk_xvector.scp $data/$utt2label_file; do
    [ ! -f $x ] && echo "$0: file $x does not exist" && exit 1;
  done

  posterior=scp:$xvector_data/xvector.scp
  if [ -f $train_post/xvector.scp ]; then
    echo "$0: we use reballanced posterior by removing training prior."
    posterior=ark:$xvector_data/norm_post
  fi

  copy-vector $posterior ark,t:- | \
    awk '{max=$3;m_ind=0; for (i=3;i<NF;i++) if($i>max) {max=$i;m_ind=i-3 };print $1,m_ind}' > $xvector_data/utt2max

elif [ -f $xvector_data/posteriors ]; then
  cat $xvector_data/posteriors | awk '{max=$3;argmax=3; for(f=3;f<NF;f++) { if ($f>max)
    { max=$f; argmax=f; }} print $1, (argmax - 3); }' > $xvector_data/utt2max
else
  echo "One of the files $xvector_data/xvector.scp or $xvector_data/posterior need to exist for evaluation."
fi

if [ ! -f $data/$utt2label_file ]; then
  echo "$0: label file $data/$utt2label_fil does not exist" && exit 1;
fi

# We map neighboring ages to single class, if amount of training data
# for some ages are less than some threshold.
# Generating map from class to minimum age correspond to class.
# generate class2age using age2class
utils/utt2spk_to_spk2utt.pl $age2class > $xvector_data/class2age || exit 1;
utils/apply_map.pl -f 2 $xvector_data/class2age < $xvector_data/utt2max > $xvector_data/utt2predicted.age

# We filter utt2label file to consider utterances that has predicted label.
# e.g. Some utterances are too short and no embedding are generated for them.
# and no label generated for them.
utils/filter_scp.pl $xvector_data/utt2predicted.age $data/$utt2label_file > $xvector_data/${utt2label_file}.filtered

paste <(sort -k1 $xvector_data/${utt2label_file}.filtered) <(sort -k1 $xvector_data/utt2predicted.age) > $xvector_data/tmp_f
awk '{print $2" "$4}' $xvector_data/tmp_f > $xvector_data/true.predicted.age
#paste <(awk '{print $2}' $data/utt2age) <(awk '{print $2}' $xvector_data/utt2predicted.age)  > $xvector_data/true.predicted.age
mean_abs_error=`cat $xvector_data/true.predicted.age | awk '{ sum +=( $1-$2 >= 0 ? $1-$2 : $2-$1 ); n++ } END { if (n > 0) print sum / n; }'`

echo "mean abs error |y_i - y'_i| is $mean_abs_error." > $xvector_data/mean_abs_error.txt

accuracy=`cat $xvector_data/true.predicted.age | awk '{ sum +=( $1==$2 ? 1 : 0 ); n++ } END { if (n > 0) print sum / n; }'`
tot_data=`wc -l $xvector_data/true.predicted.age | awk '{print $1}'`
echo "Overall accuracy is $accuracy" > $xvector_data/accuracy
num_classes=`wc -l $xvector_data/class2age | awk '{print $1}'`
for n in `cat $xvector_data/class2age | awk '{print $1}'`;do
  accuracy=`cat $xvector_data/true.predicted.age | awk -v n=$n '{if ($1 == n) print $1" "$2}' | awk -v t=$tot_data '{ sum +=( $1==$2 ? 1 : 0 ); n++ } END { if (n > 0) print sum / n", "n / t", "n; }'`
  miclassified=`cat $xvector_data/true.predicted.age | awk -v n=$n '{if ($1 == n) print $2}' | sort | uniq -c | sort -k1 -n -r | sed  's/^ *//g' | tr ' ' ':' | tr '\n' ' ,'`
  echo "accuracy, % of data, num of test data for class $n = $accuracy; (num-predicted utts with class label : class label: $miclassified)" >> $xvector_data/accuracy
done

compute-wer --mode=present --text \
  "ark:apply_map.pl -f 2 $age2class < $data/$utt2label_file |" \
  ark:$xvector_data/utt2max > $xvector_data/wer

if [ -f $data/utt2num_frames ]; then
  utils/apply_map.pl -f 3 $data/utt2num_frames < $xvector_data/tmp_f | \
    awk '{if ($2 != $4) print $1" "$2" "$4" "$3}' | sort -k 2 > $xvector_data/utt.true.predicted.len
fi


