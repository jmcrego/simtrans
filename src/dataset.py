# -*- coding: utf-8 -*-
import os.path
import io
from math import *
from random import shuffle
import random
import numpy as np
import sys
import time
import gzip
from collections import defaultdict
from tokenizer import build_tokenizer

reload(sys)
sys.setdefaultencoding('utf8')

idx_unk = 0
str_unk = "<unk>"

idx_pad = 1
str_pad = "<pad>"

idx_bos = 2
str_bos = "<bos>"

idx_eos = 3
str_eos = "<eos>"

class Embeddings():

    def __init__(self, voc, length):
        self.dim = length
        # i need an embedding for each word in voc
        # embedding matrix must have tokens in same order than voc 0:<unk>, 1:<pad>, 2:le, ...
        self.matrix = []
        for tok in voc: self.matrix.append(np.random.normal(0, 1.0, self.dim)) 
        self.matrix = np.asarray(self.matrix, dtype=np.float32)
        self.matrix = self.matrix / np.sqrt((self.matrix ** 2).sum(1))[:, None]

class Vocab():

    def __init__(self, dict_file, lid_voc=[]):
        self.tok_to_idx = {}
        self.idx_to_tok = []
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #0
        self.idx_unk = idx_unk
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #1
        self.idx_pad = idx_pad
        self.idx_to_tok.append(str_bos)
        self.tok_to_idx[str_bos] = len(self.tok_to_idx) #2
        self.idx_bos = idx_bos
        self.idx_to_tok.append(str_eos)
        self.tok_to_idx[str_eos] = len(self.tok_to_idx) #3
        self.idx_eos = idx_eos
        for lid in lid_voc:
            self.idx_to_tok.append(lid)
            self.tok_to_idx[lid] = len(self.tok_to_idx)

        nline = 0
        with io.open(dict_file, 'rb') as f:
            for line in f:
                nline += 1
                line = line.strip()
                self.idx_to_tok.append(line)
                self.tok_to_idx[line] = len(self.tok_to_idx)

        self.length = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries) {}\n'.format(self.length, dict_file))

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def exists(self, s):
        return s in self.tok_to_idx

    def get(self,s):
        if type(s) == int: ### I want the string
            if s < len(self.idx_to_tok): return self.idx_to_tok[s]
            else:
                sys.stderr.write('error: key \'{}\' not found in vocab\n'.format(s))
                sys.exit()
        ### I want the index
        if s not in self.tok_to_idx: 
            return idx_unk
        return self.tok_to_idx[s]


class Dataset():

    def __init__(self, fsrc, ftgt, config, do_shuffle):
        self.voc_src = config.voc_src 
        self.voc_tgt = config.voc_tgt
        self.lid_voc = config.lid_voc
        self.lid_add = config.lid_add
        self.fsrc = fsrc
        self.ftgt = ftgt
        self.seq_size = config.seq_size
        self.max_sents = config.max_sents
        self.do_shuffle = do_shuffle
        self.data = []
        self.data_batch = []
#        self.length = 0 ### length of the data set to be used (not necessarily the whole set)
        sys.stderr.write('Reading {} {}\n'.format(self.fsrc, self.ftgt))
        ### fsrc
        if self.fsrc.endswith('.gz'): fs = gzip.open(self.fsrc, 'rb')
        else: fs = io.open(self.fsrc, 'rb')

        ### ftgt
        if self.ftgt is not None:
            if self.ftgt.endswith('.gz'): ft = gzip.open(self.ftgt, 'rb')
            else: ft = io.open(self.ftgt, 'rb')

        ### first loop to read sentences and sort by length
        for sline in fs:

            sline = sline.strip('\n') 
            if config.tok_src: src, _ = config.tok_src.tokenize(str(sline))
            else: src = sline.split(' ')
            src.insert(0,str_bos)
            src.append(str_eos)

            tgt = []
            if self.ftgt is not None:
                tline = ft.readline().strip('\n')
                if config.tok_tgt: tgt, _ = config.tok_tgt.tokenize(str(tline))
                else: tgt = tline.split(' ')
                if self.lid_add: tgt = tgt.insert(0,self.lid_voc[0])
                tgt.append(str_eos)

            self.data.append([len(src),src,tgt])

        fs.close()
        if self.ftgt is not None: ft.close()
        self.len = len(self.data)
        sys.stderr.write('(dataset contains {} examples)\n'.format(self.len))

        ### sort by source length 
        if self.do_shuffle: 
            self.data.sort(key=lambda x: x[0])

        ### second loop to build data_batch
        batch = []
        for slen, src, tgt in self.data:
            batch.append([src, tgt])
            if len(batch)==config.batch_size:
                self.data_batch.append(batch)
                batch = []
        if len(batch): self.data_batch.append(batch)
        self.data = []

        sys.stderr.write('(dataset contains {} batches with up to {} examples each)\n'.format(len(self.data_batch), config.batch_size))


    def __iter__(self):
        self.nsents = 0
        self.nbatches = 0
        self.nsrc = 0
        self.ntgt = 0
        self.nunk_src = 0
        self.nunk_tgt = 0
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data_batch))]
        if self.do_shuffle: shuffle(indexs)
        for index in indexs:
            yield self.minibatch(index)
            self.nsents += len(self.data_batch[index])
            self.nbatches += 1
            if self.max_sents > 0 and self.nsents >= self.max_sents: break # already generated max_sents examples


    def minibatch(self, index):
        SRC, TGT, REF, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, LEN_SRC, LEN_TGT = [], [], [], [], [], [], [], [], []
        max_src, max_tgt = 0, 0
        for src, tgt in self.data_batch[index]:
            # src is like: <bos> my sentence <eos>
            # tgt is like: LID my sentence <eos>    (LID works as bos)

            self.nsrc += len(src) - 2
            nsrc_unk = 0
            isrc = []
            for s in src: 
                isrc.append(self.voc_src.get(s))
                if isrc[-1] == idx_unk: nsrc_unk += 1
            self.nunk_src += nsrc_unk
            
            ntgt_unk = 0
            iref = [] #must be: my sentence <eos>
            itgt = [] #must be: LID my sentence
            if len(tgt)>0:
                self.ntgt += len(tgt) - 2
                for i,t in enumerate(tgt): 
                    idx_t = self.voc_tgt.get(t)
                    if idx_t == idx_unk: ntgt_unk += 1
                    if i>0: iref.append(idx_t) ### all but the first element
                    if i<len(tgt)-1: itgt.append(idx_t) ### all but the last element
                self.nunk_tgt += ntgt_unk

            #### update data
            if len(isrc) > max_src: max_src = len(isrc)
            if len(itgt) > max_tgt: max_tgt = len(itgt)
            RAW_SRC.append(src)
            RAW_TGT.append(tgt)
            SRC.append(isrc)
            TGT.append(itgt)
            REF.append(iref)
            NSRC_UNK.append(nsrc_unk)
            NTGT_UNK.append(ntgt_unk)
            LEN_SRC.append(len(isrc)) ### '<bos> my sentence <eos>'
            LEN_TGT.append(len(itgt)) ### tgt: 'LID my sentence'    same length as ref: 'my sentence <bos>'

        ### add padding 
        for i in range(len(SRC)):
            while len(SRC[i]) < max_src: SRC[i].append(idx_pad) #<pad>
            while len(TGT[i]) < max_tgt: TGT[i].append(idx_pad) #<pad>
            while len(REF[i]) < max_tgt: REF[i].append(idx_pad) #<pad>

        #print("BATCH max_src={} max_tgt={}".format(max_src,max_tgt))
        #print("LEN_SRC: {}".format(LEN_SRC))
        #print("LEN_TGT: {}".format(LEN_TGT))
        #print("NSRC_UNK: {}".format(NSRC_UNK))
        #print("NTGT_UNK: {}".format(NTGT_UNK))
        #for i in range(len(SRC)):
        #    print("EXAMPLE {}/{}".format(i,len(SRC)))
        #    print("  RAW_SRC: {}".format(" ".join([e for e in RAW_SRC[i]])))
        #    print("  RAW_TGT: {}".format(" ".join([e for e in RAW_TGT[i]])))
        #    print("  SRC: {}".format(" ".join([str(e) for e in SRC[i]])))
        #    print("  TGT: {}".format(" ".join([str(e) for e in TGT[i]])))
        #    print("  REF: {}".format(" ".join([str(e) for e in REF[i]])))

        return SRC, TGT, REF, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, LEN_SRC, LEN_TGT



