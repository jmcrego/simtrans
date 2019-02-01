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

    def __init__(self, dict_file, net_lid=[]):
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

        #LID tokens used in train/valid set are also included
        for lid in net_lid:
            if lid in self.tok_to_idx:
                sys.stderr.write('error: repeated key \'{}\' in vocab (check if LIDs exist in vocab)\n'.format(lid))
                sys.exit()
            self.idx_to_tok.append(lid)
            self.tok_to_idx[lid] = len(self.tok_to_idx)

        nline = 0
        with io.open(dict_file, 'rb') as f:
            for line in f:
                nline += 1
                line = line.strip()
                if line in self.tok_to_idx:
                    sys.stderr.write('error: repeated key \'{}\' in vocab (check if LIDs exist in vocab)\n'.format(lid))
                    sys.exit()
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

    def __init__(self, fSRC, fTGT, LID, config):
        self.vocab = config.vocab
        self.max_sents = config.max_sents
        self.data = []
        self.data_batch = []
        self.maxtoks = 256

        ### check number of fSRC/fTGT/LID parameters are correct
        if len(fSRC) and len(fSRC)==len(fTGT) and len(fSRC)==len(LID): 
            self.is_inference = False ### training
            self.is_bitext = True
        elif len(fSRC) and len(fSRC)==len(fTGT) and len(LID)==0:       
            self.is_inference = True ### inference with src/tgt
            self.is_bitext = True
        elif len(fSRC) and len(fTGT)==0 and len(LID)==0:               
            self.is_inference = True ### inference with src
            self.is_bitext = False
        else:
            sys.stderr.write('error: bad number of input files {} {} {}\n'.format(len(fSRC),len(fTGT),len(LID)))
            sys.exit()

        ### check lid's exist in vocab
        for lid in LID:
            if not self.vocab.exists(lid):
                sys.stderr.write('error: LID={} does not exists in vocab\n'.format(lid))
                sys.exit()

        ###
        ### Read src/tgt file/s
        ###
        for ifile in range(len(fSRC)):
            fsrc = fSRC[ifile]
            if not os.path.exists(fsrc):
                sys.stderr.write('error: cannot find src file: {}\n'.format(fsrc))
                sys.exit()
            sys.stderr.write('Reading {}\n'.format(fsrc))
            if len(fTGT): 
                ftgt = fTGT[ifile]
                if not os.path.exists(ftgt):
                    sys.stderr.write('error: cannot find tgt file: {}\n'.format(ftgt))
                    sys.exit()
                sys.stderr.write('Reading {}\n'.format(ftgt))
            if len(LID): 
                str_lid = LID[ifile]
                sys.stderr.write('Using LID={}\n'.format(str_lid))

            ### fsrc
            if fsrc.endswith('.gz'): fs = gzip.open(fsrc, 'rb')
            else: fs = io.open(fsrc, 'rb')
            ### ftgt
            if self.is_bitext:
                if ftgt.endswith('.gz'): ft = gzip.open(ftgt, 'rb')
                else: ft = io.open(ftgt, 'rb')

            ### first loop to read sentences and sort by length
            for sline in fs:

                sline = sline.strip('\n') 
                if config.token is not None: 
                    src, _ = config.token.tokenize(str(sline))
                else: 
                    src = sline.split(' ')

                ### truncate if too long even for inference
                if len(src)>self.maxtoks: 
                    sys.stderr.write('warning: src sentence sized of {} tokens truncated to {}\n'.format(len(src),self.maxtoks))
                    src = src[0:self.maxtoks]

                src.insert(0,str_bos)
                src.append(str_eos)
                ### src is '<bos> my sentence <eos>'
                #print("src: "+" ".join([str(e) for e in src]))

                tgt = []
                if self.is_bitext: 
                    tline = ft.readline().strip('\n')
                    if config.token is not None: 
                        tgt, _ = config.token.tokenize(str(tline))
                    else: 
                        tgt = tline.split(' ')

                    ### truncate if too long even for inference
                    if len(tgt)>self.maxtoks: 
                        sys.stderr.write('warning: tgt sentence sized of {} tokens truncated to {}\n'.format(len(tgt),self.maxtoks))
                        tgt = tgt[0:self.maxtoks]

                    if self.is_inference: 
                        tgt.insert(0,str_bos)
                    else: ### is inference
                        tgt.insert(0,str_lid)
                    tgt.append(str_eos)
                ### tgt is: LID|<bos> my sentence <eos>
                #print("tgt: "+" ".join([str(e) for e in tgt]))

                if self.is_inference or config.seq_size==0 or (len(src)-2<=config.seq_size and len(tgt)-2<=config.seq_size):
                    self.data.append([len(src)-2,src,tgt])
    
            fs.close()
            if self.is_bitext: ft.close()

        self.len = len(self.data)
        sys.stderr.write('(dataset contains {} examples)\n'.format(self.len))

        ###
        ### sort by source length to allow batches with similar number of src tokens
        ###
        if not self.is_inference: 
            self.data.sort(key=lambda x: x[0])

        ###
        ### build data_batch
        ###
        batch = []
        for slen, src, tgt in self.data: # src='<bos> my sentence <eos>' tgt='LID|<bos> my stencence <eos>'
            isrc, nunk_src = self.src2isrc(src)
            #print("src\t{}".format(" ".join(str(e) for e in src))) #src <bos> Saturday , April 26 will be a workday , and in exchange , Friday , May 2 will be a day off . <eos>
            #print("isrc",isrc) #('isrc', [2, 11294, 16, 1413, 444, 53518, 17213, 13781, 0, 16, 15391, 31567, 27050, 16, 5325, 16, 8281, 340, 53518, 17213, 13781, 22502, 38251, 18, 3])
            #print("nunk_src",nunk_src) #('nunk_src', 1)
            itgt, iref, nunk_tgt = self.tgt2itgt_iref(tgt)
            ### when not is_bitext itgt, iref are [], []
            ### otherwise
            #print("tgt\t{}".format(" ".join(str(e) for e in tgt))) #tgt LIDisFrench Le Samedi 25 avril sera un jour de travail pour avoir un jour de repos le vendredi 2 mai . <eos>
            #print("itgt",itgt) #('itgt', [4, 7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18])
            #print("iref",iref) #('iref', [7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18, 3])
            #print("nunk_tgt",nunk_tgt) #('nunk_tgt', 1)
            #### when is_inference
            #tgt is: <bos> Le Samedi 25 avril sera un jour de travail pour avoir un jour de repos le vendredi 2 mai . <eos>
            #itgt is: #('itgt', [2, 7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18])
            batch.append([src, tgt, isrc, itgt, iref, nunk_src, nunk_tgt])
            if len(batch)==config.batch_size:
                self.data_batch.append(batch)
                batch = []
        if len(batch): self.data_batch.append(batch) ### last batch
        self.data = []

        sys.stderr.write('(dataset contains {} batches with up to {} examples each)\n'.format(len(self.data_batch), config.batch_size))

    def src2isrc(self, src):
        isrc = []
        nunk = 0
        for s in src:
            isrc.append(self.vocab.get(s))
            if isrc[-1] == idx_unk: nunk += 1
#            print(s,isrc[-1])
#        print("isrc:{}:".format(len(isrc)), isrc)
        return isrc, nunk

    def tgt2itgt_iref(self, tgt):
        itgt = []
        iref = []
        nunk = 0
        if self.is_bitext:
            for i,t in enumerate(tgt): 
                idx_t = self.vocab.get(t)
                if idx_t == idx_unk: nunk += 1
#                print(t,idx_t)
                if self.is_inference: #all tokens are used <bos> my sentence <eos>
                    itgt.append(idx_t)
                else: 
                    if i>0: iref.append(idx_t) ### do not include LID or <bos>
                    if i<len(tgt)-1: itgt.append(idx_t) ### do not include <eos>
#        print("itgt:{}:".format(len(itgt)), itgt)
#        print("iref:{}:".format(len(iref)), iref)
        return itgt, iref, nunk

    def __iter__(self):
        ### next are statistics of dataset
        self.nsents = 0
        self.nbatches = 0
        self.nsrc_tok = 0
        self.ntgt_tok = 0
        self.nsrc_unk = 0
        self.ntgt_unk = 0
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data_batch))]
        if not self.is_inference: shuffle(indexs)
        for index in indexs:
            yield self.minibatch(index)
            self.nsents += len(self.data_batch[index])
            self.nbatches += 1
            if self.max_sents > 0 and self.nsents >= self.max_sents: break # already generated max_sents examples


    def minibatch(self, index):
        SRC, TGT, REF, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, LEN_SRC, LEN_TGT = [], [], [], [], [], [], [], [], []
        max_src, max_tgt = 0, 0
        for src, tgt, isrc, itgt, iref, nsrc_unk, ntgt_unk in self.data_batch[index]:
            self.nsrc_tok += len(src)-2
            self.ntgt_tok += len(tgt)-2
            self.nsrc_unk += nsrc_unk
            self.ntgt_unk += ntgt_unk
            #### update max lenghts
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
            LEN_TGT.append(len(itgt)) ### both iref and itgt have the same length
        ### add padding 
        for i in range(len(SRC)):
            while len(SRC[i]) < max_src: SRC[i].append(idx_pad) #<pad>
            while len(TGT[i]) < max_tgt: TGT[i].append(idx_pad) #<pad>
            while len(REF[i]) < max_tgt: REF[i].append(idx_pad) #<pad>
#            print("TGT",TGT[i])
#            print("REF",REF[i])        
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



