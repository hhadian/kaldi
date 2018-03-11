#!/bin/bash

# Copyright 2013  Arnab Ghoshal
#                 Johns Hopkins University (author: Daniel Povey)
#           2014  Guoguo Chen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# To be run from one directory above this script.

# Begin configuration section.
weblm=
# end configuration sections

help_message="Usage: $0 [options] <train-txt> <dict> <out-dir> [fisher-dirs]
Train language models for Switchboard-1, and optionally for Fisher and \n
web-data from University of Washington.\n
options:
  --help          # print this message and exit
  --weblm DIR     # directory for web-data from University of Washington
";
lexicon=data/local/dict_chch/lexicon.txt
. utils/parse_options.sh


src=data/local/lm
dir=data/local/lm_char

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi


set -o errexit
mkdir -p $dir
export LC_ALL=C


gunzip -c $src/train.gz | python local/text_to_chars.py | gzip -c > ${dir}/train.gz
cat $src/heldout | python local/text_to_chars.py > ${dir}/heldout
gunzip -c $src/fisher/text1.gz | python local/text_to_chars.py | gzip -c > ${dir}/fisher.gz


cut -d' ' -f1 $lexicon > $dir/wordlist

# 6gram language model
ngram-count -text $dir/train.gz -order 6 -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -wbdiscount -interpolate -lm $dir/sw1.o6g.kn.gz
echo "PPL for SWBD1 6-gram LM:"
ngram -unk -lm $dir/sw1.o6g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/sw1.o6g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/6gram.ppl2


# 7gram language model
ngram-count -text $dir/train.gz -order 7 -limit-vocab -vocab $dir/wordlist \
  -unk -map-unk "<unk>" -wbdiscount -interpolate -lm $dir/sw1.o7g.kn.gz
echo "PPL for SWBD1 7-gram LM:"
ngram -unk -lm $dir/sw1.o7g.kn.gz -ppl $dir/heldout
ngram -unk -lm $dir/sw1.o7g.kn.gz -ppl $dir/heldout -debug 2 >& $dir/7gram.ppl2


mkdir -p $dir/fisher

if [ -f $dir/fisher.gz ]; then

  for x in 6 7; do
    ngram-count -text $dir/fisher.gz -order $x -limit-vocab \
      -vocab $dir/wordlist -unk -map-unk "<unk>" -wbdiscount -interpolate \
      -lm $dir/fisher/fisher.o${x}g.kn.gz
    echo "PPL for Fisher ${x}gram LM:"
    ngram -unk -lm $dir/fisher/fisher.o${x}g.kn.gz -ppl $dir/heldout
    ngram -unk -lm $dir/fisher/fisher.o${x}g.kn.gz -ppl $dir/heldout -debug 2 \
      >& $dir/fisher/${x}gram.ppl2
    compute-best-mix $dir/${x}gram.ppl2 \
      $dir/fisher/${x}gram.ppl2 >& $dir/sw1_fsh_mix.${x}gram.log
    grep 'best lambda' $dir/sw1_fsh_mix.${x}gram.log | perl -e '
      $_=<>;
      s/.*\(//; s/\).*//;
      @A = split;
      die "Expecting 2 numbers; found: $_" if(@A!=2);
      print "$A[0]\n$A[1]\n";' > $dir/sw1_fsh_mix.${x}gram.weights
    swb1_weight=$(head -1 $dir/sw1_fsh_mix.${x}gram.weights)
    fisher_weight=$(tail -n 1 $dir/sw1_fsh_mix.${x}gram.weights)
    ngram -order $x -lm $dir/sw1.o${x}g.kn.gz -lambda $swb1_weight \
      -mix-lm $dir/fisher/fisher.o${x}g.kn.gz \
      -unk -write-lm $dir/sw1_fsh.o${x}g.kn.gz
    echo "PPL for SWBD1 + Fisher ${x}gram LM:"
    ngram -unk -lm $dir/sw1_fsh.o${x}g.kn.gz -ppl $dir/heldout
  done
fi

if [ ! -z "$weblm" ]; then
  echo "Interpolating web-LM not implemented yet"
fi

## The following takes about 11 minutes to download on Eddie:
# wget --no-check-certificate http://ssli.ee.washington.edu/data/191M_conversational_web-filt+periods.gz
