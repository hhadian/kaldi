#!/usr/bin/env bash

# Copyright      2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora

# This script prepares the dictionary.

set -e
dir=data/local/dict
vocab_size=50000
. ./utils/parse_options.sh

mkdir -p $dir

# Combine Wellington corpora and replace some of their annotations
cat /export/corpora5/Wellington/WWC/SectionA.txt \
  /export/corpora5/Wellington/WWC/SectionB.txt \
  /export/corpora5/Wellington/WWC/SectionC.txt \
  /export/corpora5/Wellington/WWC/SectionD.txt \
  /export/corpora5/Wellington/WWC/SectionE.txt \
  /export/corpora5/Wellington/WWC/SectionF.txt \
  /export/corpora5/Wellington/WWC/SectionG.txt \
  /export/corpora5/Wellington/WWC/SectionH.txt \
  /export/corpora5/Wellington/WWC/SectionJ.txt \
  /export/corpora5/Wellington/WWC/SectionK.txt \
  /export/corpora5/Wellington/WWC/SectionL.txt | \
  cut -d' ' -f3- | sed "s/^[ \t]*//" > data/local/Wellington_tmp.txt

cat data/local/Wellington_tmp.txt | python3 <(
cat << EOF
import sys, io, re;
from collections import OrderedDict;
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8");
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8");
dict=OrderedDict([("^",""), ("|",""), ("_",""), ("*0",""), ("*1",""), ("*2",""), ("*3",""), ("*4",""),
  ("*5",""), ("*6",""), ("*7",""), ("*8",""), ("*9",""), ("*@","°"), ("**=",""), ("*=",""),
  ("*+$",""), ("$",""), ("*+","£"), ("*-","-"), ("*/","*"), ("*|",""), ("*{","{"), ("*}","}"),
  ("**#",""), ("*#",""), ("*?",""), ("**\"","\""), ("*\"","\""), ("**'","'"), ("*'","'"),
  ("*<",""), ("*>",""), ("**[",""), ("**]",""), ("**;",""), ("*;",""), ("**:",""), ("*:",""),
  ("\\\0",""), ("\\\15",""), ("\\\1",""), ("\\\2",""), ("\\\3",""), ("\\\6",""), ("\\\",""),
  ("{0",""), ("{15",""), ("{1",""), ("{2",""), ("{3",""), ("{6","")]);
pattern = re.compile("|".join(re.escape(key) for key in dict.keys()) + "|[^\\*]\\}");
dict["}"]="";
[sys.stdout.write(pattern.sub(lambda x: dict[x.group()[1:]] if re.match('[^\\*]\\}', x.group()) else dict[x.group()], line)) for line in sys.stdin];
EOF
) > data/local/Wellington_tmp2.txt

# First get the set of all letters that occur in data/train/text
cat data/train/text | \
  perl -ne '@A = split; shift @A; for(@A) {print join("\n", split(//)), "\n";}' | \
  sort -u > $dir/nonsilence_phones.txt

# Now use the pocolm's wordlist which is the most N frequent words in
# in data/train/text and LOB+Brown corpora (dev and test excluded) with their comprising
# letters as their transcription. Only include words that use the above letters.
# (Letter # is replaced with <HASH>)

export letters=$(cat $dir/nonsilence_phones.txt | tr -d "\n")

<<<<<<< HEAD
cut -d' ' -f2- data/train/text | \
  cat data/local/browncorpus/brown.txt - | \
  cat data/local/lobcorpus/0167/download/LOB_COCOA/lob.txt - | \
  cat data/local/Wellington_tmp2.txt - | \
=======
head -n $vocab_size data/local/local_lm/data/word_count | awk '{print $2}' | \
>>>>>>> upstream/master
  perl -e '$letters=$ENV{letters};
while(<>){
    chop;
    $w = $_;
    if($w =~ m/^[$letters]+$/){
      $trans = join(" ", split(//, $w));
      $trans =~ s/#/<HASH>/g;
      print "$w $trans\n";
    }
}' | sort -u > $dir/lexicon.txt


sed -i "s/#/<HASH>/" $dir/nonsilence_phones.txt

echo '<sil> SIL' >> $dir/lexicon.txt
echo '<unk> SIL' >> $dir/lexicon.txt

echo SIL > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
