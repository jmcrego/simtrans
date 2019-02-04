#!/usr/bin/python -u

import sys
import random
import numpy as np
from collections import defaultdict

def insert(VEC, VEC2):
#    print("INSERT")
#    print("VEC {}".format(" ".join(VEC)))
#    print("\tVEC2 {}".format(" ".join(VEC2)))
    first2 = np.random.randint(len(VEC2)-minrange+1)
    last2 = np.random.randint(first2+minrange-1,min(len(VEC2),first2+maxrange))
#    print("\t[{}, {}]".format(first2,last2))
    slice2 = VEC2[first2:last2+1]
#    print("\tslice2 {}".format(" ".join(slice2)))
    into1 = np.random.randint(len(VEC)) #[0,n)
#    print("\tinto1 {}".format(into1))
    VEC[into1:into1] = slice2 #insert slice2 into VEC at position into
#    print("\tVEC {}".format(" ".join(VEC)))
    return " ".join(VEC)

def delete(VEC):
#    print("DELETE")
#    print("VEC {}".format(" ".join(VEC)))
    first1 = np.random.randint(len(VEC)-minrange+1)
    last1 = np.random.randint(first1+minrange-1,min(len(VEC),first1+maxrange))
#    print("\t[{}, {}]".format(first1,last1))
    del VEC[first1:last1+1]
#    print("\tVEC {}".format(" ".join(VEC)))
    return " ".join(VEC)


def do_parallel(i):
    src, tgt = data[i]
    print("P\t{}\t{}".format(src, tgt))

def do_uneven(i):
    src, tgt = data[i]
    j = (i + np.random.randint(1, len(data))) % len(data)
    src2, tgt2 = data[j]
    if np.random.uniform(0, 1) >= 0.5:
        print("Us\t{}\t{}".format(src2, tgt))
    else:
        print("Ut\t{}\t{}".format(src, tgt2))

def do_insert(i):
    src, tgt = data[i]
    SRC = src.split(' ')
    TGT = tgt.split(' ')
    if len(SRC)<=minrange or len(TGT)<=minrange: return

    j = (i + np.random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    SRC2 = src2.split(' ')
    TGT2 = tgt2.split(' ')
    if len(SRC2)<=minrange or len(TGT2)<=minrange: return

    if np.random.uniform(0, 1) >= 0.5:
        print("Is\t{}\t{}".format(insert(SRC,SRC2), tgt))
    else:
        print("It\t{}\t{}".format(src, insert(TGT,TGT2)))

def do_delete(i):
    src, tgt = data[i]
    SRC = src.split(' ')
    TGT = tgt.split(' ')
    if len(SRC)<=minrange or len(TGT)<=minrange: return

    if np.random.uniform(0, 1) >= 0.5:
        print("Ds\t{}\t{}".format(delete(SRC), tgt))
    else:
        print("Dt\t{}\t{}".format(src, delete(TGT)))


#print(sorted(np.random.randint(10, size=100)))
#sys.exit()

minrange = 5
maxrange = 10
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
    if (tok=="-minrange" and len(sys.argv)):
        minrange = int(sys.argv.pop(0))
    elif (tok=="-maxrange" and len(sys.argv)):
        maxrange = int(sys.argv.pop(0))
    elif (tok=="-mode" and len(sys.argv)):
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
        if 'p' in mode: do_parallel(i)
        if 'u' in mode: do_uneven(i)
        if 'i' in mode: do_insert(i)
        if 'd' in mode: do_delete(i)


