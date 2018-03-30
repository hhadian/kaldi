#!/usr/bin/env python3
import argparse
import sys
import string 
from collections import defaultdict

parser = argparse.ArgumentParser(description="""creates lexicon.""")
parser.add_argument('nonsilence_phones_path', type=str,
                    help='Path of the phones in IAM')
parser.add_argument('corpus_data_path', type=str,
                    help='combined corpus path')
parser.add_argument('out_path', type=str,
                    help='lexicon file path')
parser.add_argument('--max_elements', type=int, default=50000,
                    help='max elements in the lexicon')
args = parser.parse_args()

max_elements = int(args.max_elements)
phones_list = []
with open(args.nonsilence_phones_path) as f:
    for line in f:
        line = line.strip()
        phones_list.append(line)


phone_set = set(phones_list)
wordcount = defaultdict(int)
text_fh = open(args.corpus_data_path)
for line in text_fh:
    for word in line.strip().split():
        word_wo_punct = word
        if word_wo_punct:
            char_set = set(word_wo_punct)
            if char_set < phone_set:
                wordcount[word_wo_punct] += 1


elements_count = 0
lexicon = dict()
lex_fh = open(args.out_path, 'w')
for key in sorted(wordcount, key=wordcount.get, reverse=True):
    elements_count += 1
    if elements_count >= max_elements:
        break
    chars = list(key)
    spaced_chars = " ".join(chars)
    spaced_chars = spaced_chars.replace("#", "<HASH>")
    lex_fh.write(key + " " + spaced_chars + "\n") 
