#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
max_elements=50000

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh || exit 1;

mkdir -p $dir
# First get the set of all letters that occur in data/train/text
cat data/train/text | \
  perl -ne '@A = split; shift @A; for(@A) {print join("\n", split(//)), "\n";}' | \
  sort -u > $dir/nonsilence_phones.txt

local/prepare_lexicon.py $dir/nonsilence_phones.txt data/local/corpus_data.txt $dir/lexicon.txt --max_elements $max_elements

sed -i "s/#/<HASH>/" $dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt
echo '<unk> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
