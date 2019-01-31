#!/usr/bin/python -u

import sys
import random
import numpy as np
from collections import defaultdict

def insert(str, str2):
    VEC = str.split(' ')
    VEC2 = str2.split(' ')
#    print("VEC {}".format(VEC))
#    print("\tVEC2 {}".format(VEC2))
    range2 = sorted(np.random.randint(len(VEC2), size=2)) #list of 2 (sorted) integers in the range [0,n)
#    print("\trange2 {}".format(range2))
    slice2 = VEC2[range2[0]:range2[1]+1]
#    print("\tslice2 {}".format(slice2))
    into1 = np.random.randint(len(VEC)) #[0,n)
#    print("\tinto1 {}".format(into1))
    VEC[into1:into1] = slice2 #insert slice2 into VEC at position into
#    print("\tVEC {}".format(VEC))
    return " ".join(VEC)

def delete(str):
    VEC = str.split(' ')
#    print("VEC {}".format(VEC))
    range1 = sorted(np.random.randint(len(VEC), size=2)) #list of 2 (sorted) integers in the range [0,n)
#    print("\trange1 {}".format(range1))
    del VEC[range1[0]:range1[1]+1]
#    print("\tVEC {}".format(VEC))
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
    j = (i + np.random.randint(1, len(data)-1)) % len(data)
    src2, tgt2 = data[j]
    if np.random.uniform(0, 1) >= 0.5:
        print("Is\t{}\t{}".format(insert(src,src2), tgt))
    else:
        print("It\t{}\t{}".format(src, insert(tgt,tgt2)))

def do_delete(i):
    src, tgt = data[i]
    if np.random.uniform(0, 1) >= 0.5:
        print("Ds\t{}\t{}".format(delete(src), tgt))
    else:
        print("Dt\t{}\t{}".format(src, delete(tgt)))


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
        if 'p' in mode: do_parallel(i)
        if 'u' in mode: do_uneven(i)
        if 'i' in mode: do_insert(i)
        if 'd' in mode: do_delete(i)


