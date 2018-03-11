#!/usr/bin/env python
# Copyright 2017   Hossein Hadian

# Apache 2.0



import sys
import math

words = []
for line in sys.stdin:
  words.append(line.strip().split()[0])

logp = -math.log(1.0 / len(words))
for w in words:
  print("0\t0\t{w}\t{w}\t{p}".format(w=w, p=logp))

print("0")
