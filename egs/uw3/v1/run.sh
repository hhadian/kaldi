#!/bin/bash

stage=0
nj=30
color=1
scale=40
pad=true
data_download=data
data_dir=data_pad
exp_dir=exp_pad

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # Data preparation
  local/prepare_data.sh --dir $data_download
fi

mkdir -p $data_dir/{train,test}/data
if [ $stage -le 1 ]; then
  for f in train test; do
    local/make_feature_vect.py --scale-size $scale --color $color $data_download/$f --pad $pad | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done
fi

numSilStates=4
numStates=8
num_gauss=10000

numLeavesTri=500
numGaussTri=20000
numLeavesMLLT=500
numGaussMLLT=20000

boost_sil=1
variance_floor_val=0.001
beam=50

lang_affix=${numStates}states_${numSilStates}sil
mono_affix=${lang_affix}_${num_gauss}_var${variance_floor_val}_beam${beam}_boost${boost_sil}
tri_affix=${mono_affix}_${numLeavesTri}_${numGaussTri}
tri2_affix=${tri_affix}_${numLeavesMLLT}_${numGaussMLLT}

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/test/ $data_dir/train/dict
  utils/prepare_lang.sh --num-sil-states $numSilStates --num-nonsil-states $numStates --position-dependent-phones false \
    $data_dir/train/dict "<sil>" $data_dir/lang_${lang_affix}/temp $data_dir/lang_${lang_affix}
fi

if [ $stage -le 3 ]; then
  cp -R $data_dir/lang_${lang_affix} -T $data_dir/lang_test_${lang_affix}
  local/prepare_lm.sh --grammar-words false $data_dir/train/text  $data_dir/lang_test_${lang_affix} 2 || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj $nj --variance_floor_val $variance_floor_val \
    --boost-silence $boost_sil --bbeam $beam \
    $data_dir/train \
    $data_dir/lang_${lang_affix} \
    $exp_dir/mono_${mono_affix}
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj \
    $data_dir/train $data_dir/lang_${lang_affix} \
    $exp_dir/mono_${mono_affix} \
    $exp_dir/mono_ali_${mono_affix}
  steps/train_deltas.sh --variance_floor_val $variance_floor_val \
    --boost-silence $boost_sil \
    $numLeavesTri $numGaussTri $data_dir/train $data_dir/lang_${lang_affix} \
    $exp_dir/mono_ali_${mono_affix} \
    $exp_dir/tri_${tri_affix}
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    $data_dir/train $data_dir/lang_${lang_affix} \
    $exp_dir/tri_${tri_affix} \
    $exp_dir/tri_ali_${tri_affix}
  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT \
    $data_dir/train $data_dir/lang_${lang_affix} \
    $exp_dir/tri_ali_${tri_affix} $exp_dir/tri2_${tri2_affix}
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test_${lang_affix} \
    $exp_dir/mono_${mono_affix} \
    $exp_dir/mono_${mono_affix}/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/mono_${mono_affix}/graph \
    $data_dir/test \
    $exp_dir/mono_${mono_affix}/decode_test
fi

if [ $stage -le 8 ]; then
  utils/mkgraph.sh $data_dir/lang_test_${lang_affix} \
    $exp_dir/tri_${tri_affix} \
    $exp_dir/tri_${tri_affix}/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/tri_${tri_affix}/graph \
    $data_dir/test \
    $exp_dir/tri_${tri_affix}/decode_test
fi

if [ $stage -le 9 ]; then
  utils/mkgraph.sh $data_dir/lang_test_${lang_affix} \
    $exp_dir/tri2_${tri2_affix} \
    $exp_dir/tri2_${tri2_affix}/graph
  steps/decode.sh --nj $nj --cmd $cmd --beam $beam \
    $exp_dir/tri2_${tri2_affix}/graph \
    $data_dir/test \
    $exp_dir/tri2_${tri2_affix}/decode_test
fi

if [ $stage -le 10 ]; then
  run_cnn_1a.sh --stage 0 \
    --data_dir ${data_dir} \
    --exp_dir ${exp_dir} \
    --gmm tri2_${tri2_affix} \
    --ali tri2_ali_${tri2_affix} \
    --lang_affix ${lang_affix}
fi
