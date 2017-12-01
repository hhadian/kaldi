#!/bin/bash

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling).

phndir=data/local/dict_phn
dir=data/local/dict_charv1
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

# Use the word list of the phoneme-based lexicon. Create the lexicon using characters.
local/swbd1_map_words.pl -f 1 $phndir/lexicon1.txt | awk '{print $1}' | \
  perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
  > $dir/lexicon1.txt

# Add special noises words & characters into the lexicon.
(echo '!sil sil'; echo '[vocalized-noise] spn'; echo '[noise] nsn'; \
  echo '[laughter] lau'; echo '<unk> spn') | \
  cat - $dir/lexicon1.txt | sort | uniq > $dir/lexicon2.txt || exit 1;

cat $dir/lexicon2.txt | sort -u > $dir/lexicon.txt || exit 1;


( echo sil; echo spn; echo nsn; echo lau ) > $dir/silence_phones.txt

cat $dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v sil > $dir/nonsilence_phones.txt  || exit 1;

echo sil > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt


echo "Character-based dictionary preparation succeeded"
