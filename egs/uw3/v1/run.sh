#!/bin/bash

stage=0
nj=30

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # Data preparation
  local/prepare_data.sh --dir data --download-dir data/download
fi

mkdir -p data/{train,test}/data
if [ $stage -le 1 ]; then
  for f in train test; do
    local/make_features.py --scale-size 40 --color 1 --pad true data/$f | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:data/$f/data/images.ark,data/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh data/$f || exit 1;
  done
fi

beam=50

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/train/ data/test/ data/train/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --position-dependent-phones false \
    data/train/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 3 ]; then
  mkdir -p data/lang_test
  cp -R data/lang/. data/lang_test/

  cp data/train/text data/train/text_copy
  cat data/test/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >test_words.txt
  cat data/train/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >train_words.txt
  filter_scp.pl --exclude train_words.txt test_words.txt >diff.txt
  cat diff.txt | awk '{ print "id " $1 }' >> data/train/text_copy

  local/prepare_lm.sh data/train/text_copy data/lang_test || exit 1;
  rm test_words.txt train_words.txt diff.txt
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --cmd $cmd \
    data/train data/lang exp/mono
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd $cmd 500 20000 \
    data/train data/lang exp/mono_ali exp/tri
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    data/train data/lang exp/tri exp/tri_ali
  steps/train_lda_mllt.sh --cmd $cmd --splice-opts "--left-context=3 --right-context=3" 500 20000 \
    data/train data/lang exp/tri_ali exp/tri2
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    exp/mono/graph data/test exp/mono/decode_test
fi

if [ $stage -le 8 ]; then
  utils/mkgraph.sh data/lang_test exp/tri exp/tri/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    exp/tri/graph data/test exp/tri/decode_test
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    exp/tri2/graph data/test exp/tri2/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd --use-graphs true \
    data/train data/lang exp/tri2 exp/tri2_ali
fi

if [ $stage -le 11 ]; then
  run_cnn_1a.sh
fi
