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

    def __init__(self, dict_file, extra_tokens = False):
        self.extra_tokens = extra_tokens #<unk> <pad> <eos> <bos>
        self.tok_to_idx = {}
        self.idx_to_tok = []
        if self.extra_tokens:
            self.idx_to_tok.append(str_unk)
            self.tok_to_idx[str_unk] = len(self.tok_to_idx) #0
            self.idx_to_tok.append(str_pad)
            self.tok_to_idx[str_pad] = len(self.tok_to_idx) #1
            self.idx_to_tok.append(str_bos)
            self.tok_to_idx[str_bos] = len(self.tok_to_idx) #2
            self.idx_to_tok.append(str_eos)
            self.tok_to_idx[str_eos] = len(self.tok_to_idx) #3
        nline = 0
        with io.open(dict_file, 'rb') as f:
            for line in f:
                nline += 1
                line = line.strip()
                self.idx_to_tok.append(line)
                self.tok_to_idx[line] = len(self.tok_to_idx)

        self.length = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries)\n'.format(self.length))

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
            if not self.extra_tokens:
                sys.stderr.write('error: key \'{}\' not found in vocab\n'.format(s))
                sys.exit()
            return idx_unk
        return self.tok_to_idx[s]


class Dataset():

    def __init__(self, fsrc, ftgt, flid, config, do_shuffle):
        self.voc_src = config.voc_src 
        self.voc_tgt = config.voc_tgt 
        self.voc_lid = config.voc_lid
        self.fsrc = fsrc
        self.ftgt = ftgt
        self.flid = flid
        self.seq_size = config.seq_size
        self.max_sents = config.max_sents
        self.do_shuffle = do_shuffle
        self.data = []
        self.length = 0 ### length of the data set to be used (not necessarily the whole set)

        ### fsrc
        if self.fsrc.endswith('.gz'): fs = gzip.open(self.fsrc, 'rb')
        else: fs = io.open(self.fsrc, 'rb')

        ### ftgt
        if self.ftgt is not None:
            if self.ftgt.endswith('.gz'): ft = gzip.open(self.ftgt, 'rb')
            else: ft = io.open(self.ftgt, 'rb')

        ### flid
        if self.flid is not None:
            if self.flid.endswith('.gz'): fl = gzip.open(self.flid, 'rb')
            else: fl = io.open(self.flid, 'rb')

        for sline in fs:

            sline = sline.strip('\n') 
            if config.tok_src: src, _ = config.tok_src.tokenize(str(sline))
            else: src = sline.split(' ')

            tgt = []
            if self.ftgt is not None:
                tline = ft.readline().strip('\n')
                if config.tok_tgt: tgt, _ = config.tok_tgt.tokenize(str(tline))
                else: tgt = tline.split(' ')

            lid = []
            if self.flid is not None:
                lline = fl.readline().strip('\n') 
                lid = [lline]

            self.data.append([src,tgt,lid])
            self.length += 1
        fs.close()
        if self.ftgt is not None: ft.close()
        if self.flid is not None: fl.close()

        if self.max_sents > 0: self.length = min(self.length,self.max_sents)
        sys.stderr.write('({} {} {} contain {} examples)\n'.format(self.fsrc, self.ftgt, self.flid,len(self.data)))

    def __len__(self):
        return self.length


    def __iter__(self):
        nsent = 0
        self.nsrc = 0
        self.ntgt = 0
        self.nunk_src = 0
        self.nunk_tgt = 0
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data))]
        if self.do_shuffle: shuffle(indexs)
        for index in indexs:
            (src, tgt, lid) = self.data[index] 
            nsrc_unk = 0
            ntgt_unk = 0

            isrc = []
            isrc.append(idx_bos)
            for s in src: 
                isrc.append(self.voc_src.get(s))
                if isrc[-1] == idx_unk: 
                    nsrc_unk += 1
                    self.nunk_src += 1
                self.nsrc += 1
            isrc.append(idx_eos)

            iref = []
            itgt = []
            if len(tgt)>0:
                itgt.append(idx_bos) ### tgt is: <bos> my sentence 
                for t in tgt: 
                    iref.append(self.voc_tgt.get(t))
                    itgt.append(self.voc_tgt.get(t))
                    if itgt[-1] == idx_unk: 
                        ntgt_unk += 1
                        self.nunk_tgt += 1
                    self.ntgt += 1
                iref.append(idx_eos) ### ref is: my sentence <eos>


            ilid = []
            for l in lid: 
                ilid.append(self.voc_lid.get(l)) ### must exist

            ### itgt, iref, ilid and lid may be empty lists
            yield isrc, itgt, iref, ilid, src, tgt, nsrc_unk, ntgt_unk
            nsent += 1
            if self.max_sents > 0 and nsent >= self.max_sents: break # already generated max_sents examples

def minibatches(data, minibatch_size):
    SRC, TGT, REF, LID, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK = [], [], [], [], [], [], [], []
    max_src, max_tgt = 0, 0
    for (src, tgt, ref, lid, raw_src, raw_tgt, nsrc_unk, ntgt_unk) in data:
        if len(SRC) == minibatch_size:
            yield build_batch(SRC, TGT, REF, LID, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, max_src, max_tgt)
            SRC, TGT, REF, LID, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK = [], [], [], [], [], [], [], []
            max_src, max_tgt = 0, 0
        if len(src) > max_src: max_src = len(src)
        if len(tgt) > max_tgt: max_tgt = len(tgt)
        SRC.append(src)
        TGT.append(tgt)
        REF.append(ref)
        LID.append(lid)
        RAW_SRC.append(raw_src)
        RAW_TGT.append(raw_tgt)
        NSRC_UNK.append(nsrc_unk)
        NTGT_UNK.append(ntgt_unk)

    if len(SRC) != 0:
        yield build_batch(SRC, TGT, REF, LID, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, max_src, max_tgt)

def build_batch(SRC, TGT, REF, LID, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, max_src, max_tgt):
    src_batch, tgt_batch, ref_batch, lid_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch = [], [], [], [], [], [], [], [], [], []
    ### build: src_batch, pad_src_batch sized of max_src
    batch_size = len(SRC)
    for i in range(batch_size):
        src = list(SRC[i])
        tgt = list(TGT[i])
        ref = list(REF[i])
        while len(src) < max_src: src.append(idx_pad) #<pad>
        while len(tgt) < max_tgt: tgt.append(idx_pad) #<pad>
        while len(ref) < max_tgt: ref.append(idx_pad) #<pad>
        ### add to batches
        len_src_batch.append(len(RAW_SRC[i]) + 2) ### added <bos> and <eos>
        len_tgt_batch.append(len(RAW_TGT[i]) + 1) ### added <eos> or <eos> (to be used for length of tgt and ref)
        src_batch.append(src)
        tgt_batch.append(tgt)
        ref_batch.append(ref)
        lid_batch.append(LID[i])
        raw_src_batch.append(RAW_SRC[i])
        raw_tgt_batch.append(RAW_TGT[i])
        nsrc_unk_batch.append(NSRC_UNK[i])
        ntgt_unk_batch.append(NTGT_UNK[i])
    return src_batch, tgt_batch, ref_batch, lid_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch


