#!/usr/bin/python3

import argparse, sys, string, io, codecs, os
from collections import defaultdict
import re

parser = argparse.ArgumentParser(description="""Creates lexicon.""")
parser.add_argument('wellington_path', type=str,
                    help='Path of the wellington corpus')
parser.add_argument('out_dir', type=str,
                    help='Where to write output file')
args = parser.parse_args()

text_fh = io.open(args.wellington_path, mode="r", encoding="utf-8")
out_fh = open(args.out_dir, 'w', encoding='ascii')
data = []
for line in text_fh:
    byte_data=line.encode("ascii","ignore")
    ascii_data=byte_data.decode('ascii')
    ascii_data_content=ascii_data.strip().split()[2:]
    ascii_data_content = " ".join(ascii_data_content)
    data.append(ascii_data_content + "\n")
    out_fh.write(ascii_data_content + "\n")

regexes = '**[', '*<*\'*4', '*<*5', '|^*4', '|^*0', '|^', '*<*4', '|^*0', '|^', '\\0', \
          '*<*4', '|^*"*0', '|^\\0', '|^*"', '**"', '*<*4', '|^*0', '|^', '|^*"', '|^\\0', \
          '*<*4', '|^*0\\0', '**]', '**\'', '*>', '*"', '|^*"*0', '*-', \
          '*- *"', '*"', '|^*"', '*0', '^', '*+$', '*<', '*1', '*0', '*\'', \
          '*<*0{0', '}', '*#', '{0'

combinedRegex = re.compile('|'.join(re.escape(x) for x in regexes))

for line in data:
    new_line = re.sub(combinedRegex, '', line)
    out_fh.write(new_line)
