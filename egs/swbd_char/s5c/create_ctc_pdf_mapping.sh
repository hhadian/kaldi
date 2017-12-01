#!/bin/bash

python -c 'import sys
sys.stdout.write("[ ")
for i in range(90): m = i/2 if i%2 == 0 else 45; sys.stdout.write(str(m) + " ");
print("]")
print("<NumPdfs> 46")
'
