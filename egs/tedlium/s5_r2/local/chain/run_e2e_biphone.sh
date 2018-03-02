#!/bin/bash


set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
affix=dim450  # 2 layers have dropout with proportion 0.2
decode_iter=
lat_beam=8.0
beam=15.0

# training options
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=12
minibatch_size=150=128,64/300=100,64,32/600=50,32,16/1200=16,8
remove_egs=false
common_egs_dir=
no_mmi_percent=20
l2_regularize=0.00005
dim=450
frames_per_iter=2500000
cmvn_opts="--norm-means=true --norm-vars=true"
leaky_hmm_coeff=0.1
hid_max_change=0.75
final_max_change=1.5
self_repair=1e-5
acwt=1.0
post_acwt=10.0
num_scale_opts="--transition-scale=1.0 --self-loop-scale=1.0"
equal_align_iters=19
den_use_initials=true
den_use_finals=false
slc=1.0
shared_phones=true
train_set=train_cleaned_seg_spEx_hires
topo_affix=_chain
tree_affix=_sharedT1S1
topo_opts=
uniform_lexicon=false
first_layer_splice=-1,0,1
add_deltas=false
disable_ng=false
momentum=0
nnet_block=relu-batchnorm-layer
no_viterbi_percent=100
src_tree_dir=  # set this in case we want to use another tree (this won't be end2end anymore)
drop_prop=0.2
drop_schedule=
combine_sto_penalty=0.0
dbl_chk=true
prefinal_dim=$dim
frame_subsampling_factor=3
normalize_egs=false
n_tie=0
use_final_stddev=true
max_dur_opts=

proportional_shrink=0.0
chunk_left_context=0
chunk_right_context=0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5 ; echo '')
printf "\n################ $rid #################\n" >> drop_runs.log
echo "run-id: $rid" >> drop_runs.log
echo `date` >> drop_runs.log
echo "$0 $@"  >> drop_runs.log

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

lang=data/lang_e2e${topo_affix}
treedir=exp/chain/e2e_bitree${tree_affix}_topo${topo_affix}
dir=exp/chain/e2e_nodrop_biphone${affix}${topo_affix}${tree_affix}
echo "Run $rid, dir = $dir" >> drop_runs.log

input_dim=40
if $add_deltas; then
  input_dim=120
fi

#local/nnet3/run_e2e_common.sh --stage $stage \
#  --speed-perturb $speed_perturb \
#  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo_e2e.py $topo_opts \
                                    $nonsilphonelist $silphonelist >$lang/topo
#  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  steps/nnet3/chain/prepare_e2e_biphone.sh --nj 30 --cmd "$train_cmd" \
                                           --shared-phones $shared_phones \
                                           --uniform-lexicon $uniform_lexicon \
                                           --scale-opts "$num_scale_opts" \
                                           --treedir "$src_tree_dir" \
                                           data/$train_set $lang $treedir
  cp exp/chain/e2e_base/phone_lm.fst $treedir/
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  final_stddev=0
  if $use_final_stddev; then
    final_stddev=$(echo "print(1.0/$dim)" | python)
  fi

  if $disable_ng; then
    common="affine-comp=AffineComponent"
  fi
  lc1=3
  rc1=3
  lc2=3
  rc2=3
  if [ "$frame_subsampling_factor" == "2" ]; then
    echo "------------------- Frame subsampling factor is 2 -------------------"
    lc1=2
    rc1=2
    lc2=4
    rc2=4
  fi
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  #learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=$input_dim name=input

  # the first splicing is moved before the lda layer, so no splicing here
  $nnet_block name=tdnn1 input=Append($first_layer_splice) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn2 input=Append(-1,0,1) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn3 input=Append(-1,0,1,2) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn4 input=Append(-$lc1,0,$rc1) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn5 input=Append(-$lc2,0,$rc1) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn6 input=Append(-$lc2,0,$rc2) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common

  $nnet_block name=prefinal-chain input=tdnn6 dim=$prefinal_dim target-rms=$final_layer_normalize_target self-repair-scale=$self_repair $common
  output-layer name=output include-log-softmax=true dim=$num_targets max-change=$final_max_change $common param-stddev=$final_stddev

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

#    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
  num_phone_sets=$(cat $lang/phones/sets.int | wc -l)
  num_pdfs_per_phone=1
  if [[ $topo_opts == *"2pdf"* ]]; then
    num_pdfs_per_phone=2
  fi
  steps/nnet3/chain/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient $leaky_hmm_coeff \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --chain.frame-subsampling-factor=$frame_subsampling_factor \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--normalize-egs $normalize_egs --add-deltas $add_deltas" \
    --trainer.options="--offset_first_transitions=$normalize_egs --den-use-initials=$den_use_initials --den-use-finals=$den_use_finals --check-derivs=$dbl_chk" \
    --trainer.dropout-schedule "$drop_schedule" \
    --trainer.no-mmi-percent $no_mmi_percent \
    --trainer.n-tie $n_tie \
    --trainer.tie-info "--num-phone-sets=$num_phone_sets --num-pdfs-per-phone=$num_pdfs_per_phone" \
    --trainer.no-viterbi-percent $no_viterbi_percent \
    --trainer.max-dur-opts "$max_dur_opts" \
    --trainer.equal-align-iters $equal_align_iters \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.combine-sum-to-one-penalty $combine_sto_penalty \
    --trainer.optimization.momentum $momentum \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.optimization.proportional-shrink $proportional_shrink \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 10 \
    --cleanup false \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi


if [ $stage -le 14 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
fi

if [ $stage -le 15 ]; then
  tie_opts=
  if [ "$n_tie" != "0" ]; then
    tie_opts="--pdf-map-filename=$dir/pdf-map.txt"
  fi

  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj 30 --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nnet3-latgen-opts "$tie_opts" \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset}_hires $dir/decode_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

echo "Run $rid done. Date: $(date). Results:" >> drop_runs.log
local/chain/compare_wer_general.sh $dir
local/chain/compare_wer_general.sh $dir >> drop_runs.log
