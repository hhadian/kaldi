
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lex", help="path to data/lang/words.txt")
parser.add_argument("text", help="path to data/test/text")

args = parser.parse_args()


lex = set()
with open(args.lex) as f:
    for l in f:
        w = l.strip().split()[0]
        lex.add(w.lower())

num_not_found = 0
num_tot = 0
with open(args.text) as f:
    for l in f:
        words = l.strip().split()[1:]
        for w in words:
            num_tot +=1
            if w.lower() not in lex:
                num_not_found += 1

print('Total test words: {}, total OOVs: {}, '
      'OOV rate: {}'.format(num_tot, num_not_found, num_not_found * 1.0 / num_tot))
