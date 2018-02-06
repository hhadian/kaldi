#!/bin/bash


echo -n "System               "
for x in $*; do   printf "% 10s" $x;   done
echo

#for d in exp/sdm1/chain_cleaned/tdnn*/decode_*; do grep Sum $d/*sc*/*ys | utils/best_wer.sh; done|grep eval_hires


echo -n "WER on dev  "
for x in $*; do
  wer=$(grep Sum ${x}/decode_dev*/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "WER on eval  "
for x in $*; do
  wer=$(grep Sum ${x}/decode_eval*/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Final train prob     "
for x in $*; do
  prob=$(grep Overall ${x}/log/compute_prob_train.final.log | grep -v xent | awk '{print $8}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob     "
for x in $*; do
  prob=$(grep Overall ${x}/log/compute_prob_valid.final.log | grep -v xent | awk '{print $8}')
  printf "% 10s" $prob
done
echo

echo -n "Final train prob (xent)    "
for x in $*; do
  prob=$(grep Overall ${x}/log/compute_prob_train.final.log | grep -w xent | awk '{print $8}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob (xent)    "
for x in $*; do
  prob=$(grep Overall ${x}/log/compute_prob_valid.final.log | grep -w xent | awk '{print $8}')
  printf "% 10s" $prob
done
echo
