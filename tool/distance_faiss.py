# -*- coding: utf-8 -*-
import sys
import faiss
import numpy as np
import time
import datetime

fdb = None
fquery = None
c = 0
K = 1
s = 0.0
gpu = False
usage = """usage: {} -db FILE -query FILE [-c INT] [-k INT] [-s FLOAT] [-gpu] [-h]
   -db    FILE : file with sentences and their corresponding vector representations
   -query FILE : file with sentences and their corresponding vector representations
   -c      INT : column containing vector representations (starting by 0) [0]
   -k      INT : show the k nearest sentences [1]
   -s    FLOAT : minimum similarity to consider two sentences near [0.0]
   -gpu        : use gpu (passed through CUDA_VISIBLE+DEVICES)
   -h          : this help
This scripts finds in file -fdb the -k nearest sentences of sentences in file -fquery. 
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

def read_embeddings(file,c):
    t1 = time.time()
    sys.stderr.write('Reading db: {} '.format(file))
    f = open(file,"r")
    emb = []
    n = 0
    for l in f:
        n += 1
        if n%100000 == 0:
            if n%1000000 == 0: sys.stderr.write(str(n))
            else: sys.stderr.write(".")
        ls = l.strip().split("\t")[c].split()
        emb.append([float(i) for i in ls])
    tend = time.time()
    sys.stderr.write("[{} sentences, {:.2f} seconds]".format(len(emb),tend-tini))
    emb = np.array(emb).astype('float32')
    emb = emb/np.sqrt(np.sum(emb*emb,1))[:,None]
    t2 = time.time()
    sys.stderr.write("[NORMALIZED {:.2f} seconds]\n".format(t2-t1))
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
    return index

#################################
### main
#################################

emb_db = read_embeddings(fdb,c)
emb_query = read_embeddings(fquery,c)

start = time.time()
index = db_indexs()
index.add(emb_db)
distances, index_ = index.search(emb_query, 10) #emb_db.shape[0])
for i in range(index_.shape[0]):
    for j in range(10):
        print("{:.6f}\t{}\t{}".format(distances[i,j], i, index_[i,j]))
end = time.time()
sys.stderr.write("selecting rate: {:.2f} sentences per second\n".format(index_.shape[0]/(end-start)))
