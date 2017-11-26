#!/usr/bin/env bash

""" This module prepares dictionary directory. It creates lexicon.txt,
    silence_phones.txt, optional_silence.txt and extra_questions.txt.  
    
    Args:
     $1: location of train tet file.
     $2: location of test text file.
     $3: location of dict directory.

    Eg. local/prepare_dict.sh data/train/ data/test/ data/train/dict
"""

train_text=$1
test_text=$2
dir=$3

mkdir -p $dir

local/prepare_lexicon.py $train_text $test_text $dir

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

( echo '<sil> SIL'; ) >> $dir/lexicon.txt || exit 1;
( echo '<unk> SIL'; ) >> $dir/lexicon.txt || exit 1;

( echo SIL ) > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
