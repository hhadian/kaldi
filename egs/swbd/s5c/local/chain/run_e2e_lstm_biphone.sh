#!/bin/bash


# TO TRY: full set (no nodup)
set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
affix=_notset
decode_iter=
lat_beam=6.5
beam=15.0

# training options
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
#minibatch_size=150=32/300=20,8/600=10/1200=4
minibatch_size=150=100,64/300=50,32/600=28,16/1200=8
remove_egs=true
common_egs_dir=
no_mmi_percent=0
l2_regularize=0.00005
dim=800
frames_per_iter=2500000
cmvn_opts="--norm-means=true --norm-vars=true"
leaky_hmm_coeff=0.1
hid_max_change=0.75
final_max_change=1.5
self_repair=1e-5
acwt=1.0
post_acwt=10.0
num_scale_opts="--transition-scale=1.0 --self-loop-scale=1.0"
equal_align_iters=0
den_use_initials=true
den_use_finals=false
slc=1.0
shared_phones=true
train_set=train_nodup_seg_spEx_hires
topo_affix=_chain
tree_affix=_sharedT1S1
topo_opts="--type chain"
uniform_lexicon=false
first_layer_splice=-1,0,1
add_deltas=false
disable_ng=false
momentum=0
nnet_block=relu-batchnorm-layer
no_viterbi_percent=100
src_tree_dir=  # set this in case we want to use another tree (this won't be end2end anymore)
drop_schedule=
drop_prop=0.2
combine_sto_penalty=0.0
dbl_chk=true
n_tie=0
normalize_egs=true
cmd=queue.pl
use_final_stddev=false

lang_decode=data/lang_sw1_tg
lang_affix=
has_fisher=true
test_sets="train_dev_hires eval2000_hires"

chunk_left_context=40
chunk_right_context=0
extra_left_context=50
extra_right_context=0
label_delay=5

# End configuration section.
echo "$0 $@"  # Print the command line for logging

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5 ; echo '')
printf "\n################ $rid #################\n" >> tlstm_runs.log
echo "run-id: $rid" >> tlstm_runs.log
echo `date` >> tlstm_runs.log
echo "$0 $@"  >> tlstm_runs.log

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

drop_affix=
if [ ! -z $drop_schedule ]; then
  drop_affix="_drop_2Ls"
fi

lang=data/lang_e2e${topo_affix}
treedir=exp/chain/e2e_bitree${tree_affix}_topo${topo_affix}
dir=exp/chain/e2e${drop_affix}_tlstm_biphone${affix}${topo_affix}${tree_affix}
echo "Run $rid, dir = $dir" >> tlstm_runs.log

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

  if $disable_ng; then
    common="affine-comp=AffineComponent"
  fi

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  #learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  pdim=$[dim/4]
  final_stddev=0
  if $use_final_stddev; then
    final_stddev=$(echo "print(1.0/$dim)" | python)
  fi

  mkdir -p $dir/configs
  if [ -z $drop_schedule ]; then
  cat <<EOF > $dir/configs/network.xconfig

  input dim=$input_dim name=input

  $nnet_block name=tdnn1 input=Append($first_layer_splice) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  $nnet_block name=tdnn2 input=Append(-1,0,1) dim=$dim
  $nnet_block name=tdnn3 input=Append(-1,0,1) dim=$dim

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts
  $nnet_block name=tdnn4 input=Append(-3,0,3) dim=$dim
  $nnet_block name=tdnn5 input=Append(-3,0,3) dim=$dim
  fast-lstmp-layer name=fastlstm2 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts
  $nnet_block name=tdnn6 input=Append(-3,0,3) dim=$dim
  $nnet_block name=tdnn7 input=Append(-3,0,3) dim=$dim
  fast-lstmp-layer name=fastlstm3 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts

  output-layer name=output include-log-softmax=true dim=$num_targets max-change=$final_max_change $common param-stddev=$final_stddev

EOF

  else
  cat <<EOF > $dir/configs/network.xconfig

  input dim=$input_dim name=input

  $nnet_block name=tdnn1 input=Append($first_layer_splice) dim=$dim max-change=$hid_max_change self-repair-scale=$self_repair $common
  relu-batchnorm-dropout-layer name=tdnn2 input=Append(-1,0,1) dropout-proportion=$drop_prop dim=$dim
  $nnet_block name=tdnn3 input=Append(-1,0,1) dim=$dim

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts
  $nnet_block name=tdnn4 input=Append(-3,0,3) dim=$dim
  $nnet_block name=tdnn5 input=Append(-3,0,3) dim=$dim
  fast-lstmp-layer name=fastlstm2 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts
  relu-batchnorm-dropout-layer name=tdnn6 input=Append(-3,0,3) dropout-proportion=$drop_prop dim=$dim
  $nnet_block name=tdnn7 input=Append(-3,0,3) dim=$dim
  fast-lstmp-layer name=fastlstm3 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$pdim delay=-3 $lstm_opts

  output-layer name=output include-log-softmax=true dim=$num_targets max-change=$final_max_change $common param-stddev=$final_stddev

EOF
  fi
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  num_phone_sets=$(cat $lang/phones/sets.int | wc -l)
  num_pdfs_per_phone=1
  if [[ $topo_opts == *"2pdf"* ]]; then
    num_pdfs_per_phone=2
  fi
  steps/nnet3/chain/train_e2e.py --stage $train_stage \
    --cmd "$cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient $leaky_hmm_coeff \
    --chain.l2-regularize $l2_regularize \
    --trainer.dropout-schedule "$drop_schedule" \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--normalize-egs $normalize_egs --add-deltas $add_deltas --num-train-egs-combine 800" \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.options="--offset_first_transitions=$normalize_egs --den-use-initials=$den_use_initials --den-use-finals=$den_use_finals --check-derivs=$dbl_chk" \
    --trainer.no-mmi-percent $no_mmi_percent \
    --trainer.no-viterbi-percent $no_viterbi_percent \
    --trainer.equal-align-iters $equal_align_iters \
    --trainer.n-tie $n_tie \
    --trainer.tie-info "--num-phone-sets=$num_phone_sets --num-pdfs-per-phone=$num_pdfs_per_phone" \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.deriv-truncate-margin 8 \
    --trainer.optimization.momentum $momentum \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 10 \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi


graph_dir=$dir/graph_${lang_affix}
if [ $stage -le 14 ]; then
  rm -rf $dir/graph_sw1_tg
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale $slc $lang_decode $dir $graph_dir
fi

#          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \

decode_suff=sw1_tg
wtstr="_a${acwt}_p${post_acwt}_s${slc}"
wtstr=
if [ $stage -le 15 ]; then
  tie_opts=
  if [ "$n_tie" != "0" ]; then
    tie_opts="--pdf-map-filename=$dir/pdf-map.txt"
  fi

  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in $test_sets; do
      (
      rm -r $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${lang_affix}_${decode_suff} || true
      steps/nnet3/decode.sh --acwt $acwt --post-decode-acwt $post_acwt --num-threads 4 \
                            --nnet3-latgen-opts "$tie_opts" \
                            --lattice-beam $lat_beam --beam $beam \
                            --extra-left-context $extra_left_context  \
                            --extra-right-context $extra_right_context  \
                            --extra-left-context-initial 0 \
                            --extra-right-context-final 0 \
                            --frames-per-chunk 300 \
                            --nj 40 --cmd "$decode_cmd" $iter_opts \
                            --add-deltas $add_deltas \
        $graph_dir data/${decode_set} $dir/decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_${lang_affix}_${decode_suff} || exit 1;
#      ln -sf decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}
      ~/bin/swbd.sh $dir
      if $has_fisher; then
        rm -r $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${lang_affix}_sw1_fsh_fg || true
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set} \
          $dir/decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_${lang_affix}_sw1_{tg,fsh_fg} || exit 1;
#        ln -sf decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_sw1_fsh_fg $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_fsh_fg
      fi
      ) &
  done
fi
wait;

~/bin/swbd.sh $dir
echo "Run $rid done. Date: $(date). Results:" >> tlstm_runs.log
~/bin/swbd.sh $dir >> tlstm_runs.log
local/chain/compare_wer_general.sh $dir
local/chain/compare_wer_general.sh $dir >> tlstm_runs.log
