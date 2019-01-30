#!/usr/bin/python -u

import sys
#from random import randint
import random
import numpy as np
from collections import defaultdict

def do_parallel(i):
    src, tgt = data[i]
    print("{}\t{}\t{}".format('P', src, tgt))

def do_uneven(i):
    src, tgt = data[i]
    j = (i + random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    if random.uniform(0, 1) >= 0.5:
        print("{}\t{}\t{}".format('U', src2, tgt))
    else:
        print("{}\t{}\t{}".format('U', src, tgt2))

def do_insert(i):
    src, tgt = data[i]
    j = (i + random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    if random.uniform(0, 1) >= 0.5:
        SRC2 = src2.split(' ')
        first2 = random.randint(0,len(SRC2)) #[0,n)
        last2 = random.randint(first2,len(SRC2)) #[first,n)
        SRC = src.split(' ')
        into = random.randint(0,len(SRC)) #[0,n)
        SRC.insert(into," ".join(SRC2[first2:last2+1]))
        print("{}{}\t{}\t{}".format('Is',last2-first2+1, " ".join(SRC), tgt))
    else:
        TGT2 = tgt2.split(' ')
        first2 = random.randint(0,len(TGT2)) #[0,n)
        last2 = random.randint(first2,len(TGT2)) #[first,n)
        TGT = tgt.split(' ')
        into = random.randint(0,len(TGT)) #[0,n)
        TGT.insert(into," ".join(TGT2[first2:last2+1]))   
        print("{}{}\t{}\t{}".format('It',last2-first2+1, src, " ".join(TGT)))

def do_delete(i):
    src, tgt = data[i]
    if random.uniform(0, 1) >= 0.5:
        SRC2 = src.split(' ')
        first2 = random.randint(0,len(SRC2)) #[0,n)
        last2 = random.randint(first2,len(SRC2)) #[first,n)
        SRC = SRC2[0:first2-1] + SRC2[last2+1:]
        print("{}{}\t{}\t{}".format('Ds',last2-first2+1, " ".join(SRC), tgt))
    else:
        TGT2 = tgt.split(' ')
        first2 = random.randint(0,len(TGT2)) #[0,n)
        last2 = random.randint(first2,len(TGT2)) #[first,n)
        TGT = TGT2[0:first2-1] + TGT2[last2+1:]
        print("{}{}\t{}\t{}".format('Dt',last2-first2+1, src, " ".join(TGT)))


do_threshold = False
do_generate = False
usage = """usage: {} [-mode STRING] [-threshold] < INPUT
   -mode   STRING : use p:parallel u:uneven i:insert d:delete to generate sentence pairs
   -threshold     : optimizes similarity threshold
   -h             : this help
This script either:
- Generates src/tgt similar/non-similar translations according to -mode options (previous to tune threshold).
- Optimizes the similarity threshold 
INPUT must be, either:
- ^src tgt$ (source and target sentence separated by tab)
- ^ref pre$ (reference and predicted similarities separated by tab)
""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if (tok=="-mode" and len(sys.argv)):
        do_generate = True
        mode = sys.argv.pop(0)
    elif (tok=="-threshold"):
        do_threshold = True
    elif (tok=="-h"):
        sys.stderr.write("{}\n".format(usage))
        sys.exit()
    else:
        sys.stderr.write('error: unparsed {} option\n{}\n'.format(tok,usage))
        sys.exit()

data = []
for line in sys.stdin:
    (src, tgt) = line.rstrip('\n').split('\t')
    data.append([src,tgt])
sys.stderr.write("Read {} entries\n".format(len(data)))

if do_generate:
    for i in range(len(data)):
        do_parallel(i)
        do_uneven(i)
        do_insert(i)
        do_delete(i)


