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
from vocab import Vocab

reload(sys)
sys.setdefaultencoding('utf8')

class Dataset():

    def __init__(self, fSRC, fTGT, LID, config):
        self.vocab_src = config.vocab_src
        self.vocab_tgt = config.vocab_tgt
        self.max_sents = config.max_sents
        self.data = []
        self.data_batch = []
        self.maxtoksperline = 512
        self.is_inference = config.is_inference

        if len(fSRC) == len(fTGT):
            self.is_bitext = True 
        else:
            self.is_bitext = False

        ### check lid's exist in vocab
        for lid in LID:
            lid = self.vocab_src.beg_delim + lid + self.vocab_src.end_delim
            if not self.vocab_src.exists(lid):
                sys.stderr.write('error: LID={} does not exists in vocab_src\n'.format(lid))
                sys.exit()
            if not self.vocab_tgt.exists(lid):
                sys.stderr.write('error: LID={} does not exists in vocab_src\n'.format(lid))
                sys.exit()


        if (len(LID) and len(fSRC)!=len(LID)) or (len(fTGT) and len(fSRC)!=len(fTGT)):
            sys.stderr.write('error: bad number of input files {} {} {}\n'.format(len(fSRC),len(fTGT),len(LID)))
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
                str_lid = self.vocab_src.beg_delim + LID[ifile] + self.vocab_src.end_delim
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
                if config.token_src is not None: 
                    src, _ = config.token_src.tokenize(str(sline))
                else: 
                    src = sline.split(' ')

                ### truncate if too long even for inference
                if len(src)>self.maxtoksperline and self.is_inference: 
                    sys.stderr.write('warning: src sentence sized of {} tokens truncated to {}\n'.format(len(src),self.maxtoksperline))
                    src = src[0:self.maxtoksperline]

                tgt = []
                if self.is_bitext: 
                    tline = ft.readline().strip('\n')
                    if config.token_tgt is not None: 
                        tgt, _ = config.token_tgt.tokenize(str(tline))
                    else: 
                        tgt = tline.split(' ')

                    ### truncate if too long even for inference
                    if len(tgt)>self.maxtoksperline and self.is_inference: 
                        sys.stderr.write('warning: tgt sentence sized of {} tokens truncated to {}\n'.format(len(tgt),self.maxtoksperline))
                        tgt = tgt[0:self.maxtoksperline]

                #print("src\t{}".format(" ".join(str(e) for e in src))) 
                #print("tgt\t{}".format(" ".join(str(e) for e in tgt))) 

                if self.is_inference or (len(src)<=config.max_seq_size and len(tgt)<=config.max_seq_size):
                    self.data.append([len(src),src,tgt,str_lid])
    
            fs.close()
            if self.is_bitext: ft.close()

        self.len = len(self.data)
        sys.stderr.write('(dataset contains {} examples)\n'.format(self.len))

        ###
        ### sort by source length to allow batches with similar number of src tokens
        ###
        sys.stderr.write('(sorting examples)\n')
        if not self.is_inference: 
            self.data.sort(key=lambda x: x[0])

        ###
        ### build data_batch
        ###
        sys.stderr.write('(building batches)\n')
        batch = []
        prev_tgt = []
        n_divergent = 0
        for slen, src, tgt, lid in self.data: # src='my sentence' tgt='my sentence <lid>'
            sign = -1.0 ### not divergent
            if not self.is_inference and config.network.ali is not None and len(prev_tgt)>0 and np.random.random_sample()>=0.5: #### divergent example
                tgt = prev_tgt ### delete <bos> and <eos> added previously
                n_divergent += 1
                sign = 1.0

            isrc, div_src, nunk_src = self.src2isrc_div(src,sign)
            itgt, div_tgt, iwrd, iref, nunk_tgt = self.tgt2itgt_div_iwrd_iref(tgt,sign,lid) #if not bitext, returns [], [], [], 0 
            prev_tgt = tgt[1:-1]

            #print("{}\nsrc\t{}".format(slen," ".join(str(e) for e in src))) 
            #print("isrc\t{}".format(" ".join(str(e) for e in isrc))) 
            #print("div_src\t{}".format(" ".join(str(e) for e in div_src))) 
            #print("nunk_src\t{}".format(nunk_src))
            #
            #print("tgt\t{}".format(" ".join(str(e) for e in tgt))) 
            #print("itgt\t{}".format(" ".join(str(e) for e in itgt))) 
            #print("div_tgt\t{}".format(" ".join(str(e) for e in div_tgt))) 
            #print("nunk_tgt\t{}".format(nunk_tgt))
            #print("iwrd\t{}".format(" ".join(str(e) for e in iwrd))) 
            #print("iref\t{}".format(" ".join(str(e) for e in iref))) 

            batch.append([src, tgt, isrc, itgt, iwrd, iref, div_src, div_tgt, nunk_src, nunk_tgt])
            if len(batch)==config.batch_size:
                self.data_batch.append(batch)
                batch = []
        if len(batch): self.data_batch.append(batch) ### last batch
        self.data = []

        sys.stderr.write('(dataset contains {} batches with up to {} examples each. {} divergent examples)\n'.format(len(self.data_batch), config.batch_size, n_divergent))


    def src2isrc_div(self, src, sign):
        src = list(src)
        src.insert(0,self.vocab_src.str_bos)
        src.append(self.vocab_src.str_eos)
        #src is    [ <bos>, src1, src2, ..., srcj, <eos>]
        isrc = [] #[ <BOS>, SRC1, SRC2, ..., SRCj, <EOS>]
        div = []  #[ sign,  sign, sign, ..., sign, sign]
        nunk = 0
        for s in src:
            div.append(sign) # sign is -1.0 if not divergent +1.0 if divergent
            isrc.append(self.vocab_src.get(s))
            if isrc[-1] == self.vocab_src.idx_unk: 
                nunk += 1
        return isrc, div, nunk

    def tgt2itgt_div_iwrd_iref(self, tgt, sign, lid):
        tgt = list(tgt)
        tgt.insert(0,self.vocab_tgt.str_bos)
        tgt.append(self.vocab_tgt.str_eos)
        #tgt is    [ <bos>, tgt1, tgt2, ..., tgti, <eos>]
        itgt = [] #[ <BOS>, TGT1, TGT2, ..., TGTi, <EOS>]
        div =  [] #[ sign,  sign, sign, ..., sign, sign]
        iwrd = [] #[ <lid>, TGT1, TGT2, ..., TGTi]  (removes last element <eos>, replaces <BOS> by <LID>)
        iref = [] #[ TGT1,  TGT2, ..., TGTi, <EOS>] (removes first element <bos>)
        nunk = 0
        for t in tgt:
            div.append(sign) # sign is -1.0 if not divergent +1.0 if divergent
            itgt.append(self.vocab_tgt.get(t))
            if itgt[-1] == self.vocab_tgt.idx_unk: 
                nunk += 1

        iwrd = list(itgt)
        del iwrd[-1] #deleted <eos>
        iwrd[0] = self.vocab_tgt.get(lid) #<bos> => <lid>
        iref = list(itgt)
        del iref[0] #deleted <bos>
        return itgt, div, iwrd, iref, nunk



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
        SRC, TGT, ISRC, ITGT, IWRD, IREF, DIV_SRC, DIV_TGT, NUNK_SRC, NUNK_TGT, LEN_SRC, LEN_TGT, LEN_WRD = [], [], [], [], [], [], [], [], [], [], [], [], []
        max_src, max_tgt, max_ref = 0, 0, 0
        for src, tgt, isrc, itgt, iwrd, iref, div_src, div_tgt, nsrc_unk, ntgt_unk in self.data_batch[index]:
            self.nsrc_tok += max(0,len(src))
            self.ntgt_tok += max(0,len(tgt))
            self.nsrc_unk += nsrc_unk
            self.ntgt_unk += ntgt_unk
            SRC.append(src)
            TGT.append(tgt)
            ISRC.append(isrc)
            ITGT.append(itgt)
            IWRD.append(iwrd)
            IREF.append(iref)
            DIV_SRC.append(div_src)
            DIV_TGT.append(div_tgt)
            LEN_SRC.append(len(isrc)) ### len(isrc)
            LEN_TGT.append(len(itgt)) ### len(itgt)
            LEN_WRD.append(len(iref)) ### len(iwrd) and len(iref)
            max_src =  max(max_src, LEN_SRC[-1])
            max_ref =  max(max_ref, LEN_WRD[-1])
            max_tgt =  max(max_tgt, LEN_TGT[-1])
        #print("BATCH max_src={} max_tgt={} max_ref={}".format(max_src,max_tgt,max_ref))
        for i in range(len(SRC)):
            ### add padding 
            while len(ISRC[i]) < max_src: ISRC[i].append(self.vocab_src.idx_pad) #<pad>
            while len(ITGT[i]) < max_tgt: ITGT[i].append(self.vocab_tgt.idx_pad) #<pad>
            while len(DIV_SRC[i]) < max_src: DIV_SRC[i].append(1.0)
            while len(DIV_TGT[i]) < max_tgt: DIV_TGT[i].append(1.0)
            while len(IWRD[i]) < max_ref: IWRD[i].append(self.vocab_tgt.idx_pad) #<pad>
            while len(IREF[i]) < max_ref: IREF[i].append(self.vocab_tgt.idx_pad) #<pad>
            #print(" SRC\t{}".format(" ".join(str(e) for e in SRC[i]))) 
            #print(" ISRC\t{}".format(" ".join(str(e) for e in ISRC[i]))) 
            #print(" LEN_SRC\t{}".format(LEN_SRC[i])) 
            #print(" TGT\t{}".format(" ".join(str(e) for e in TGT[i]))) 
            #print(" ITGT\t{}".format(" ".join(str(e) for e in ITGT[i]))) 
            #print(" LEN_TGT\t{}".format(LEN_TGT[i]))
            #print(" IWRD\t{}".format(" ".join(str(e) for e in IWRD[i]))) 
            #print(" IREF\t{}".format(" ".join(str(e) for e in IREF[i]))) 
            #print(" LEN_WRD\t{}".format(LEN_WRD[i]))
            #print(" DIV_SRC\t{}".format(" ".join(str(e) for e in DIV_SRC[i]))) 
            #print(" DIV_TGT\t{}".format(" ".join(str(e) for e in DIV_TGT[i]))) 
        return ISRC, ITGT, IWRD, IREF, DIV_SRC, DIV_TGT, SRC, TGT, LEN_SRC, LEN_TGT, LEN_WRD



