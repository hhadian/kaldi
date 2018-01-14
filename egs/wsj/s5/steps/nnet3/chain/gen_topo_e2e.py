#!/usr/bin/env python

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#                                     2017 Hossein Hadian

# This file is as ./gen_topo.py used to be (before we extended the transition-model
# code to support having a different self-loop pdf-class).  It is included
# here for baseline and testing purposes.


# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.  This is a modified version of
# 'utils/gen_topo.pl' that generates a different type of topology, one that we
# believe should be useful in the 'chain' model.  Note: right now it doesn't
# have any real options, and it treats silence and nonsilence the same.  The
# intention is that you write different versions of this script, or add options,
# if you experiment with it.

from __future__ import print_function
import argparse
import sys

parser = argparse.ArgumentParser(description="Usage: steps/nnet3/chain/gen_topo.py "
                                             "<colon-separated-nonsilence-phones> <colon-separated-silence-phones>"
                                             "e.g.:  steps/nnet3/chain/gen_topo.pl 4:5:6:7:8:9:10 1:2:3\n",
                                 epilog="See egs/swbd/s5c/local/chain/train_tdnn_a.sh for example of usage.");
parser.add_argument("nonsilence_phones", type=str,
                    help="List of non-silence phones as integers, separated by colons, e.g. 4:5:6:7:8:9");
parser.add_argument("silence_phones", type=str,
                    help="List of silence phones as integers, separated by colons, e.g. 1:2:3");
parser.add_argument("--sil-self-loop-prob", type=float, default=0.5)
parser.add_argument("--nonsil-self-loop-prob", type=float, default=0.5)
parser.add_argument("--type", type=str, choices=['1pdf', '2pdf', '3pdf', '3pdfloop', '2pdfloop', 'chain', 'ctc'], default="chain")


args = parser.parse_args()

silence_phones = [ int(x) for x in args.silence_phones.split(":") ]
nonsilence_phones = [ int(x) for x in args.nonsilence_phones.split(":") ]
all_phones = silence_phones +  nonsilence_phones
sil_p = args.sil_self_loop_prob
nonsil_p = args.nonsil_self_loop_prob


print("<Topology>")

if args.type == "chain":
    assert(sil_p == nonsil_p)
    assert(sil_p == 0.5)
    print("<TopologyEntry>")
    print("<ForPhones>")
    print(" ".join([str(x) for x in all_phones]))
    print("</ForPhones>")
    print("<State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1 <Transition> 0 {} <Transition> 1 {} </State>".format(sil_p, 1.0 - sil_p))
    print("<State> 1 </State>")
    print("</TopologyEntry>")
elif args.type == "3pdf":
    print("<TopologyEntry>")
    print("<ForPhones>")
    print(" ".join([str(x) for x in all_phones]))
    print("</ForPhones>")
    print("<State> 0 <PdfClass> 0 <Transition> 1 {} <Transition> 3 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 1 <PdfClass> 1 <Transition> 2 {} <Transition> 3 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 2 <PdfClass> 2 <Transition> 2 {} <Transition> 3 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 3 </State>")
    print("</TopologyEntry>")
elif args.type == "3pdfloop":
    print("<TopologyEntry>")
    print("<ForPhones>")
    print(" ".join([str(x) for x in all_phones]))
    print("</ForPhones>")
    print("<State> 0 <PdfClass> 0 <Transition> 0 {} <Transition> 1 {} <Transition> 3 {} </State>".format(0.5, 0.5, 0.5))
    print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} <Transition> 3 {} </State>".format(0.5, 0.5, 0.5))
    print("<State> 2 <PdfClass> 2 <Transition> 2 {} <Transition> 3 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 3 </State>")
    print("</TopologyEntry>")
elif args.type == "ctc":
    print("<TopologyEntry>")
    print("<ForPhones>")
    print(" ".join([str(x) for x in all_phones]))
    print("</ForPhones>")
    print("<State> 0 <PdfClass> 0 <Transition> 0 {} <Transition> 1 {} <Transition> 2 {} </State>".format(0.5, 0.5, 0.5))
    print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 2 </State>")
    print("</TopologyEntry>")
elif args.type == "2pdfloop":
    print("<TopologyEntry>")
    print("<ForPhones>")
    print(" ".join([str(x) for x in all_phones]))
    print("</ForPhones>")
    print("<State> 0 <PdfClass> 0 <Transition> 0 {} <Transition> 1 {} <Transition> 2 {} </State>".format(0.33, 0.34, 0.33))
    print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
    print("<State> 2 </State>")
    print("</TopologyEntry>")
elif args.type == "2pdf":
    if sil_p == nonsil_p:
        print("<TopologyEntry>")
        print("<ForPhones>")
        print(" ".join([str(x) for x in all_phones]))
        print("</ForPhones>")
        print("<State> 0 <PdfClass> 0 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
        print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
        print("<State> 2 </State>")
        print("</TopologyEntry>")
    else:
        print("<TopologyEntry>")
        print("<ForPhones>")
        print(" ".join([str(x) for x in silence_phones]))
        print("</ForPhones>")
        print("<State> 0 <PdfClass> 0 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
        print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} </State>".format(sil_p, 1.0-sil_p))
        print("<State> 2 </State>")
        print("</TopologyEntry>")
        print("<TopologyEntry>")
        print("<ForPhones>")
        print(" ".join([str(x) for x in nonsilence_phones]))
        print("</ForPhones>")
        print("<State> 0 <PdfClass> 0 <Transition> 1 {} <Transition> 2 {} </State>".format(nonsil_p, 1.0-nonsil_p))
        print("<State> 1 <PdfClass> 1 <Transition> 1 {} <Transition> 2 {} </State>".format(nonsil_p, 1.0-nonsil_p))
        print("<State> 2 </State>")
        print("</TopologyEntry>")
else:
    if sil_p == nonsil_p:
        print("<TopologyEntry>")
        print("<ForPhones>")
        print(" ".join([str(x) for x in all_phones]))
        print("</ForPhones>")
        print("<State> 0 <PdfClass> 0 <Transition> 0 {} <Transition> 1 {} </State>".format(sil_p, 1.0-sil_p))
        print("<State> 1 </State>")
        print("</TopologyEntry>")
    else:
        sys.stderr.write("Error: Not supported.")
        sys.exit(1)

print("</Topology>")
