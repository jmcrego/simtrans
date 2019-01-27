# -*- coding: utf-8 -*-
import sys
import faiss
import numpy as np
import time
import datetime

fdb = None
fquery = None
K = 1
s = 0.0
parallel = False
nbests = False
no_normalize = False
gpu = False
usage = """usage: {} -db FILE -query FILE [-k INT] [-s FLOAT] [-gpu] [-parallel] [-no_normalize] [-nbests] [-h]
   -db    FILE  : file with sentences and their corresponding vector representations
   -query FILE  : file with sentences and their corresponding vector representations
   -k      INT  : show the k nearest sentences [1]
   -s    FLOAT  : minimum similarity to consider two sentences near [0.0]
   -parallel    : output accuracy (files must be parallel)
   -nbests      : output n-best results
   -no_normalize: do not normalize vectors
   -gpu         : use gpu (passed through CUDA_VISIBLE_DEVICES)
   -h           : this help
This scripts finds in file -db the -k nearest sentences to those in file -query. 
Distance is computed as inner product of the vectors representing each sentence pair.""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if (tok=="-db" and len(sys.argv)):
        fdb = sys.argv.pop(0)
    elif (tok=="-query" and len(sys.argv)):
        fquery = sys.argv.pop(0)
    elif (tok=="-k" and len(sys.argv)):
        K = int(sys.argv.pop(0))
    elif (tok=="-c" and len(sys.argv)):
        c = int(sys.argv.pop(0))
    elif (tok=="-s" and len(sys.argv)):
        s = float(sys.argv.pop(0))
    elif (tok=="-gpu"):
        gpu = True
    elif (tok=="-parallel"):
        parallel = True
    elif (tok=="-no_normalize"):
        normalize = True
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

def read_embeddings(file):
    t1 = time.time()
    sys.stderr.write('Reading embeddings: {} '.format(file))
    f = open(file,"r")
    emb = []
    n = 0
    for l in f:
        n += 1
        if n%100000 == 0:
            if n%1000000 == 0: sys.stderr.write(str(n))
            else: sys.stderr.write(".")
        ls = l.strip().split()
        emb.append([float(i) for i in ls])
    emb = np.array(emb).astype('float32')
    t2 = time.time()
    sys.stderr.write("[{} sentences, {:.2f} seconds]".format(len(emb),t2-t1))
    if not no_normalize: 
        emb = emb/np.sqrt(np.sum(emb*emb,1))[:,None]
        t3 = time.time()
        sys.stderr.write("[NORMALIZED {:.2f} seconds]\n".format(t3-t2))
    return emb

def db_indexs(gpu, emb_db):
    if gpu and hasattr(faiss, 'StandardGpuResources'):
        ngpus = faiss.get_num_gpus()
        sys.stderr.write("number of GPUs: {}\n".format(ngpus))
        if ngpus>=1:
            cpu_index = faiss.IndexFlatIP(emb_db.shape[1]) #IP
            index = faiss.index_cpu_to_all_gpus(cpu_index)    
        else:
            resource = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device=2
            index = faiss.GpuIndexFlatIP(resource, emb_db.shape[1], config)
    else:
        sys.stderr.write("cpu mode\n")
        index = faiss.IndexFlatIP(emb_db.shape[1])
    index.add(emb_db)
    return index

#################################
### main
#################################

emb_db = read_embeddings(fdb)
emb_query = read_embeddings(fquery)
start = time.time()
index = db_indexs(gpu, emb_db)

distances, index_ = index.search(emb_query, K)
for i in range(index_.shape[0]):
    for j in range(K):
        print("{:.6f}\t{}\t{}".format(distances[i,j], i, index_[i,j]))
end = time.time()
sys.stderr.write("selecting rate: {:.2f} sentences per second\n".format(index_.shape[0]/(end-start)))
