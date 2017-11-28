#!/usr/bin/env python

""" This module will be used by scripts for open vocabulary setup.
 If the hypothesis transcription contains <unk>, then it will replace the 
 <unk> with the word predicted by <unk> model by concatenating phones decoded 
 from the unk-model. It is currently supported only for triphone setup.

 Args:
  phones: File name of a file that contains the phones.txt, (symbol-table for phones).
          phone and phoneID, Eg. a 217, phoneID of 'a' is 217. 
  words: File name of a file that contains the words.txt, (symbol-table for words). 
         word and wordID. Eg. ACCOUNTANCY 234, wordID of 'ACCOUNTANCY' is 234.
  unk: ID of <unk>. Eg. 231.
  1best-arc-post: A file in arc-post format, which is a list of timing info and posterior
               of arcs along the one-best path from the lattice.
               E.g. 506_m01-049-00 8 12  1 7722  282 272 288 231
                    <utterance-id> <start-frame> <num-frames> <posterior> <word> [<ali>] 
                    [<phone1> <phone2>...]
  output-text: File containing hypothesis transcription with <unk> recognized by the
               unk-model.
               E.g. A move to stop mr. gaitskell.
  
  Eg. local/unk_arc_post_to_transcription.py lang/phones.txt lang/words.txt 
      data/lang/oov.int

"""
import argparse
import os
import sys
parser = argparse.ArgumentParser(description="""uses phones to convert unk to word""")
parser.add_argument('phones', type=str, help='File name of a file that contains the'
                    'symbol-table for phones. Each line must be: <phone> <phoneID>')
parser.add_argument('words', type=str, help='File name of a file that contains the'
                    'symbol-table for words. Each line must be: <word> <word-id>')
parser.add_argument('unk', type=str, default='-', help='File name of a file that'
                    'contains the ID of <unk>. The content must be: <oov-id>, e.g. 231')
parser.add_argument('--1best-arc-post', type=str, default='-', help='A file in arc-post'
                    'format, which is a list of timing info and posterior of arcs'
                    'along the one-best path from the lattice')
parser.add_argument('--output-text', type=str, default='-', help='File containing'
                    'hypothesis transcription with <unk> recognized by the unk-model')
args = parser.parse_args()

### main ###
phone_fh = open(args.phones, 'r') #create file handles 
word_fh = open(args.words, 'r')
unk_fh = open(args.unk,'r')
if args.1best_arc_post == '-':
    input_fh = sys.stdin
else:
    input_fh = open(args.1best_arc_post,'r')
if args.output_text == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.output_text,'wb')

phone_dict = dict() #stores mapping from phoneID(int) to phone(char)
phone_data_vect = phone_fh.read().strip().split("\n")

for key_val in phone_data_vect:
  key_val = key_val.split(" ")
  phone_dict[key_val[1]] = key_val[0]

word_dict = dict()
word_data_vect = word_fh.read().strip().split("\n")

for key_val in word_data_vect:
  key_val = key_val.split(" ")
  word_dict[key_val[1]] = key_val[0]
unk_val = unk_fh.read().strip().split(" ")[0]

utt_word_dict = dict() #dict of list, stores mapping from utteranceID(int) to words(str)
for line in input_fh:
  line_vect = line.strip().split("\t")
  if len(line_vect) < 6: #check for 1best-arc-post output
    print "Error: Invalid line in the 1best-arc-post file"
    print line_vect
    continue
  uttID = line_vect[0]
  word = line_vect[4]
  phones = line_vect[5]
  if uttID not in utt_word_dict.keys():
    utt_word_dict[uttID] = list()

  if word == unk_val: #Get the 1best phone sequence given by the unk-model
    phone_id_seq = phones.split(" ")
    phone_seq = list()
    for pkey in phone_id_seq:
      phone_seq.append(phone_dict[pkey]) #Convert the phone-id sequence to a phone sequence.
    phone_2_word = list()
    for phone_val in phone_seq:
      phone_2_word.append(phone_val.split('_')[0]) # removing the world-position markers(e.g. _B)
    phone_2_word = ''.join(phone_2_word) #concatnate phone sequence
    utt_word_dict[uttID].append(phone_2_word) #store word from unk-model
  else:
    if word == '0': #store space/silence
      word_val = ' '
    else:
      word_val = word_dict[word]
    utt_word_dict[uttID].append(word_val) #store word from 1best-arc-post

transcription = "" #output transcription
for utt_key in sorted(utt_word_dict.iterkeys()):
  transcription = utt_key
  for word in utt_word_dict[utt_key]:
    transcription = transcription + " " + word
  out_fh.write(transcription + '\n')
