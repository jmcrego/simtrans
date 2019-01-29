#!/usr/bin/python -u

import sys
import numpy as np
from collections import defaultdict

class nearest:
    def __init__(self, fdb, no_normalize):
        ### read tst sentences
        self.normalize = not no_normalize
        self.VEC = []
        with open(fdb) as f:
            idb = 0
            for line in f:
                vec = np.array(map(float, line.rstrip('\n').split(' ')))
                if self.normalize:
                    vec = vec/np.linalg.norm(vec)
                self.VEC.append(vec)
                idb += 1
        sys.stderr.write('Read db file:{} with {} embeddings\n'.format(fdb,len(self.VEC)))

    def query(self, fquery, no_normalize, K, s, parallel, nbests):
        ### read trn sentences
        nok = 0 ### used if -parallel
        iquery = 0
        acc = 0.0
        with open(fquery) as f:
            for line in f:
                vec = np.array(map(float, line.rstrip('\n').split(' ')))
                if self.normalize: 
                    vec = vec/np.linalg.norm(vec)
                ### find proximity to all db sentences
                res = defaultdict(float)
                for idb in range(len(self.VEC)):
                    sim = np.sum(self.VEC[idb] * vec) 
                    res[idb] = sim

                ### output the tst sentences closest to this trn sentence
                k = 0
                for idb in sorted(res, key=res.get, reverse=True):    
                    sim = res[idb]
                    if sim < s: 
                        break
                    if parallel and iquery==idb: 
                        nok += 1
                    if nbests:
                        print("{:.5f}\t{}\t{}".format(sim,iquery,idb))
                    k += 1
                    if k == K: 
                        break
    
                iquery += 1

            if parallel: 
                acc = 100.0*nok/len(self.VEC)
        sys.stderr.write('Processed query file:{} with {} embeddings\n'.format(fquery,iquery))
        return acc

f = None
K = 1
s = 0.0
parallel = False
nbests = False
no_normalize = False
usage = """usage: {} -db FILE -query FILE [-k INT] [-s FLOAT] [-parallel] [-nbests] [-no_normalize] [-h] 
   -db     FILE : file with sentences and their corresponding vector representations
   -query  FILE : file with sentences and their corresponding vector representations
   -k       INT : show the k nearest sentences [1]
   -s     FLOAT : minimum similarity to consider two sentences near [0.0]
   -parallel    : output accuracy (files must be parallel)
   -nbests      : output n-best results
   -no_normalize: do not normalize vectors
   -h           : this help
This scripts finds in file -fdb the -k nearest sentences to each sentence in fquery file
with a similarity score lower than -s. Similarity is computed as the cosine distance of 
the vectors representing each sentence pair.""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if (tok=="-db" and len(sys.argv)):
        fdb = sys.argv.pop(0)
    elif (tok=="-query" and len(sys.argv)):
        fquery = sys.argv.pop(0)
    elif (tok=="-k" and len(sys.argv)):
        K = int(sys.argv.pop(0))
    elif (tok=="-s" and len(sys.argv)):
        s = float(sys.argv.pop(0))
    elif (tok=="-parallel"):
        parallel = True
    elif (tok=="-no_normalize"):
        no_normalize = True
    elif (tok=="-nbests"):
        nbests = True
    elif (tok=="-h"):
        sys.stderr.write("{}\n".format(usage))
        sys.exit()
    else:
        sys.stderr.write('error: unparsed {} option\n{}\n'.format(tok,usage))
        sys.exit()

if fdb is None:
    sys.stderr.write('error: missing -db option\n{}\n'.format(usage))
    sys.exit()

if fquery is None:
    sys.stderr.write('error: missing -query option\n{}\n'.format(usage))
    sys.exit()

if not parallel and not nbests:
    sys.stderr.write('error: missing either -parallel or -nbests options\n{}\n'.format(usage))
    sys.exit()

db = nearest(fdb,no_normalize)
acc = db.query(fquery,no_normalize,K,s,parallel,nbests)

if parallel:
    print("Acc = {:.2f} %".format(acc))

