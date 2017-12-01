#!/bin/bash


set -e

# configs for 'chain'
affix=
stage=12
train_stage=-10
get_egs_stage=-10
decode_iter=
has_fisher=true

# training options
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=150=64,32/300=32,16,8/600=16,8/1200=8,4
remove_egs=true
common_egs_dir=
no_mmi_percent=101
l2_regularize=0.00005
tdnn_dim=500
lstm_dim=500
frames_per_iter=2000000
cmvn_opts="--norm-means=true --norm-vars=true"
leaky_hmm_coeff=0.1
momentum=0
chunk_left_context=40
chunk_right_context=40
self_repair_scale=0.00001
label_delay=0

acwt=1.0
post_acwt=10.0
num_scale_opts="--transition-scale=1.0 --self-loop-scale=1.0"
equal_align_iters=5
den_use_initials=true
den_use_finals=false
slc=1.0
shared_phones=true
train_set=train_nodup_seg_spEx_hires
topo_affix=_1pdf
tree_affix=_sharedT1S1
topo_opts="--type 1pdf"
uniform_lexicon=false
first_layer_splice=0
disable_ng=false
final_layer_normalize_target=0.5
dbl_chk=false
normalize_egs=true
use_final_stddev=true

# End configuration section.
echo "$0 $@"  # Print the command line for logging

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5 ; echo '')
printf "\n################ $rid #################\n" >> blstm_runs.log
echo "run-id: $rid" >> blstm_runs.log
echo `date` >> blstm_runs.log
echo "$0 $@"  >> blstm_runs.log

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
treedir=exp/chain/e2e_tree${tree_affix}_topo${topo_affix}
dir=exp/chain/e2eblstm${affix}${topo_affix}${tree_affix}
echo "Run $rid, dir = $dir" >> blstm_runs.log

nonproj_dim=$[lstm_dim/4]
proj_dim=$[lstm_dim/4]
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
  steps/nnet3/chain/prepare_e2e.sh --nj 30 --cmd "$train_cmd" \
                                   --shared-phones $shared_phones \
                                   --uniform-lexicon $uniform_lexicon \
                                   --scale-opts "$num_scale_opts" \
                                   data/$train_set $lang $treedir
  cp exp/chain/e2e_tree_a/phone_lm.fst $treedir/
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')

  # recurrent-projection-dim=$proj_dim non-recurrent-projection-dim=$nonproj_dim
  lstm_opts="decay-time=20 lstm-nonlinearity-options=\" max-change=0.75 self-repair-scale=$self_repair_scale\""
  dnn_opts=
  if $disable_ng; then
    lstm_opts="$lstm_opts affine-comp=AffineComponent"
    dnn_opts="affine-comp=AffineComponent"
  fi
  final_stddev=0
  if $use_final_stddev; then
    final_stddev=$(echo "print(1.0/$tdnn_dim)" | python)
  fi

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  relu-batchnorm-layer name=tdnn1 input=Append($first_layer_splice) dim=$tdnn_dim self-repair-scale=$self_repair_scale $dnn_opts

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstm-layer name=blstm1-forward input=tdnn1 cell-dim=$lstm_dim delay=-3 $lstm_opts
  fast-lstm-layer name=blstm1-backward input=tdnn1 cell-dim=$lstm_dim delay=3 $lstm_opts

  fast-lstm-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) cell-dim=$lstm_dim delay=-3 $lstm_opts
  fast-lstm-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) cell-dim=$lstm_dim delay=3 $lstm_opts

  fast-lstm-layer name=blstm3-forward input=Append(blstm2-forward, blstm2-backward) cell-dim=$lstm_dim delay=-3 $lstm_opts
  fast-lstm-layer name=blstm3-backward input=Append(blstm2-forward, blstm2-backward) cell-dim=$lstm_dim delay=3 $lstm_opts

  relu-batchnorm-layer name=dnn1 input=Append(blstm3-forward, blstm3-backward) dim=$tdnn_dim target-rms=$final_layer_normalize_target self-repair-scale=$self_repair_scale $dnn_opts
  output-layer name=output input=dnn1 output-delay=$label_delay include-log-softmax=true dim=$num_targets max-change=1.5 $dnn_opts param-stddev=$final_stddev

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

#    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \

  steps/nnet3/chain/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient $leaky_hmm_coeff \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--normalize-egs $normalize_egs" \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --trainer.options="--compiler.cache-capacity=512 --offset-first-transitions=$normalize_egs --den-use-initials=$den_use_initials --den-use-finals=$den_use_finals --check-derivs=$dbl_chk" \
    --trainer.no-mmi-percent $no_mmi_percent \
    --trainer.equal-align-iters $equal_align_iters \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.optimization.momentum $momentum \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 10 \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;

fi
#mv $dir/final.mdl $dir/final_wop.mdl; nnet3-am-adjust-priors $dir/final_wop.mdl $dir/priors.vec $dir/final.mdl

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale $slc data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
wtstr="_a${acwt}_p${post_acwt}_s${slc}"
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 15 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=300;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
    (
      rm -r $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || true
      steps/nnet3/decode.sh --acwt $acwt --post-decode-acwt $post_acwt \
                            --nj 50 --cmd "$decode_cmd" $iter_opts \
                            --extra-left-context $extra_left_context  \
                            --extra-right-context $extra_right_context  \
                            --frames-per-chunk "$frames_per_chunk" \
                            $graph_dir data/${decode_set}_hires \
                            $dir/decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;

      ln -sf \
         decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} \
         $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}
      ~/bin/swbd.sh $dir
      echo "HAS FISHERRRRRRRRRRRRRRRRRRRRRRRRRRRR: $has_fisher"
      if $has_fisher; then
        echo "LR RESCORRRRRING"
        rm -r $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_fsh_fg || true
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
                                      data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
                                      $dir/decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
        ln -sf decode${wtstr}_${decode_set}${decode_iter:+_$decode_iter}_sw1_fsh_fg \
           $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_fsh_fg
      fi
      ) &
  done
fi
wait;

~/bin/swbd.sh $dir
echo "Run $rid done. Date: $(date). Results:" >> blstm_runs.log
~/bin/swbd.sh $dir >> blstm_runs.log
local/chain/compare_wer_general.sh $dir
local/chain/compare_wer_general.sh $dir >> blstm_runs.log
