#!/usr/bin/python -u

import sys
import numpy as np
from collections import defaultdict

f = None
c_emb = 0
c_txt = None
K = 1
s = 0.0
parallel = False
usage = """usage: {} -f FILE [-k INT] [-s FLOAT] [-h] < SENTENCES
   -f    FILE : file with sentences and their corresponding vector representations
   -c_emb INT : column containing vector representations (starting by 0) [0]
   -c_txt INT : column containing textual sentences (starting by 0) []
   -k     INT : show the k nearest sentences [1]
   -s   FLOAT : minimum similarity to consider two sentences near [0.0]
   -parallel  : sentences are parallel
   -h         : this help
This scripts finds in file -f the -k nearest sentences to each sentence in 
SENTENCES and with a similarity score lower than -s. Similarity is computed 
as the cosine distance of the vectors representing each sentence pair.""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if (tok=="-f" and len(sys.argv)):
        f = sys.argv.pop(0)
    elif (tok=="-k" and len(sys.argv)):
        K = int(sys.argv.pop(0))
    elif (tok=="-c_emb" and len(sys.argv)):
        c_emb = int(sys.argv.pop(0))
    elif (tok=="-c_txt" and len(sys.argv)):
        c_txt = int(sys.argv.pop(0))
    elif (tok=="-s" and len(sys.argv)):
        s = float(sys.argv.pop(0))
    elif (tok=="-parallel"):
        parallel = True
    elif (tok=="-h"):
        sys.stderr.write("{}\n".format(usage))
        sys.exit()
    else:
        sys.stderr.write('error: unparsed {} option\n{}\n'.format(tok,usage))
        sys.exit()

if f is None:
    sys.stderr.write('error: missing -f option\n{}\n'.format(usage))
    sys.exit()

### read tst sentences
VEC = []
TXT = []
with open(f) as f:
    nline = 0
    for line in f:
        tok = line.rstrip('\n').split('\t')
        vec = map(float, tok[c_emb].strip().split(' '))
        vec = vec/np.linalg.norm(vec)
        VEC.append(vec)
        if c_txt is not None: TXT.append(tok[c_txt])
        nline += 1

### read trn sentences
nok = 0 ### used if -parallel
nline = 0
for line in sys.stdin:
    tok = line.rstrip('\n').split('\t')
    vec = map(float, tok[c_emb].strip().split(' '))
    vec = vec/np.linalg.norm(vec)
#    if c_txt is not None: 
#        print ("{}\t{}".format(nline, tok[c_txt]))
#    else:
#        print ("{}".format(nline))

    ### find proximity to all tst sentences
    res = defaultdict(float)
    for i in range(len(VEC)):
        sim = np.sum(VEC[i] * vec) 
        res[i] = sim

    ### output the tst sentences closest to this trn sentence
    k = 0
    for i in sorted(res, key=res.get, reverse=True):    
        sim = res[i]
        if sim < s: break
        if parallel:
            if nline==i: nok += 1
        else:
            if c_txt is not None: 
                print("{:.5f}\t{}\t{}\t{}\t{}".format(sim,nline,tok[c_txt],i,TXT[i]))
            else:
                print("{:.5f}\t{}\t{}".format(sim,nline,i))
        k += 1
        if k == K: break
    
    nline += 1

if parallel:
    print("Acc = {:.2f} %".format(100*nok/nline))

