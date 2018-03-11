import sys

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    chars = []
    for w in words:
        if w in ['[noise]', '[vocalized-noise]', '[laughter]', '<unk>']:
            chars.append(w)
        else:
            chars += list(w + '@')
    print(' '.join(chars))
