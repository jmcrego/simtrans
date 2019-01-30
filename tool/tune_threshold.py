#!/usr/bin/python -u

import sys
#from random import randint
import random
import numpy as np
from collections import defaultdict

def do_parallel(i):
    src, tgt = data[i]
    print("P\t{}\t{}".format(src, tgt))

def do_uneven(i):
    src, tgt = data[i]
    j = (i + random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    if random.uniform(0, 1) >= 0.5:
        print("Us\t{}\t{}".format(src2, tgt))
    else:
        print("Ut\t{}\t{}".format(src, tgt2))

def do_insert(i):
    src, tgt = data[i]
    j = (i + random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    if random.uniform(0, 1) >= 0.5:
#        print("src",src)
#        print("src2",src2)
        SRC2 = src2.split(' ')
#        print("len(SRC2)",len(SRC2))
        first2 = random.randint(0,len(SRC2)-1) #[0,n)
        last2 = random.randint(first2,len(SRC2)-1) #[first,n)
#        print("first2",str(first2))
#        print("last2",str(last2))
        SRC = src.split(' ')
#        print("len(SRC)",len(SRC))
        into = random.randint(0,len(SRC)) #[0,n)
#        print("into",str(into))
        SRC.insert(into," ".join(SRC2[first2:last2+1]))
        print("Is_{}_{}\t{}\t{}".format(into, last2-first2+1, " ".join(SRC), tgt))
    else:
        TGT2 = tgt2.split(' ')
        first2 = random.randint(0,len(TGT2)-1) #[0,n)
        last2 = random.randint(first2,len(TGT2)-1) #[first,n)
        TGT = tgt.split(' ')
        into = random.randint(0,len(TGT)) #[0,n)
        TGT.insert(into," ".join(TGT2[first2:last2+1]))   
        print("It_{}_{}\t{}\t{}".format(into, last2-first2+1, src, " ".join(TGT)))

def do_delete(i):
    src, tgt = data[i]
    if random.uniform(0, 1) >= 0.5:
        SRC2 = src.split(' ')
        first2 = random.randint(0,len(SRC2)) #[0,n)
        last2 = random.randint(first2,len(SRC2)) #[first,n)
        SRC = SRC2[0:first2-1] + SRC2[last2+1:]
        print("Ds_{}_{}\t{}\t{}".format(first2,last2, " ".join(SRC), tgt))
    else:
        TGT2 = tgt.split(' ')
        first2 = random.randint(0,len(TGT2)) #[0,n)
        last2 = random.randint(first2,len(TGT2)) #[first,n)
        TGT = TGT2[0:first2-1] + TGT2[last2+1:]
        print("Dt_{}_{}\t{}\t{}".format(first2,last2, src, " ".join(TGT)))


do_threshold = False
do_generate = False
seed = 1234
usage = """usage: {} [-mode STRING] [-threshold] < INPUT
   -mode   STRING : use p:parallel u:uneven i:insert d:delete to generate sentence pairs
   -threshold     : optimizes similarity threshold
   -seed      INT : seed [1234]
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
    elif (tok=="-seed" and len(sys.argv)):
        seed = int(sys.argv.pop(0))
    elif (tok=="-h"):
        sys.stderr.write("{}\n".format(usage))
        sys.exit()
    else:
        sys.stderr.write('error: unparsed {} option\n{}\n'.format(tok,usage))
        sys.exit()

random.seed(seed)
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


