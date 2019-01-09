#!/usr/bin/python -u

import sys
import numpy as np
from collections import defaultdict

f = None
K = 1
s = 0.0
usage = """usage: {} -f FILE [-k INT] [-s FLOAT] [-h] < SENTENCES
   -f  FILE : file with sentences and their corresponding vector representations
   -k   INT : show the k nearest sentences [1]
   -s FLOAT : minimum similarity to consider two sentences near [0.0]
   -h       : this help
This scripts finds in file -f the -k nearest sentences to each sentence in 
SENTENCES and with a similarity score lower than -s. Similarity is computed 
as the cosine distance of the vectors representing each sentence pair.""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if (tok=="-f" and len(sys.argv)):
        f = sys.argv.pop(0)
    elif (tok=="-k" and len(sys.argv)):
        K = int(sys.argv.pop(0))
    elif (tok=="-s" and len(sys.argv)):
        s = float(sys.argv.pop(0))
    elif (tok=="-h"):
        sys.stderr.write("{}\n".format(usage))
        sys.exit()
    else:
        sys.stderr.write('error: unparsed {} option\n{}\n'.format(tok,usage))
        sys.exit()

if f is None:
    sys.stderr.write('error: missing -f option\n{}\n'.format(usage))
    sys.exit()

VEC = []
TXT = []
with open(f) as f:
    for line in f:
        txt, vtxt = line.rstrip('\n').split('\t')
        vec = map(float, vtxt.strip().split(' '))
        vec = vec/np.linalg.norm(vec)
        VEC.append(vec)
        TXT.append(txt)

nline = 0
for line in sys.stdin:
    nline += 1
    txt, vtxt = line.rstrip('\n').split('\t')
    vec = map(float, vtxt.strip().split(' '))
    vec = vec/np.linalg.norm(vec)
    res = defaultdict(float)
    for i in range(len(VEC)):
        sim = np.sum(VEC[i] * vec) 
        res[i] = sim
    print ("{}\t{}".format(nline, txt))
    k = 0
    for i in sorted(res, key=res.get, reverse=True):    
        sim = res[i]
        if sim < s: break
        print("\t{:.4f}\t{}\t{}".format(sim,i+1,TXT[i]))
        k += 1
        if k == K: break
