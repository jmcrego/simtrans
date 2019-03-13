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
        self.vocab = config.vocab
        self.max_sents = config.max_sents
        self.max_seq_size = config.max_seq_size
        self.data = []
        self.data_batch = []
        self.maxtoksperline = 512
        self.is_inference = False
        self.is_bitext = False
        self.is_align = config.network.type=='align'

        ### check number of fSRC/fTGT/LID parameters are correct
        if len(fSRC) and len(fSRC)==len(fTGT) and len(fSRC)==len(LID): 
            self.is_bitext = True ### training
        elif len(fSRC) and len(fSRC)==len(fTGT) and len(LID)==0:       
            self.is_inference = True ### inference with src/tgt
            self.is_bitext = True
        elif len(fSRC) and len(fTGT)==0 and len(LID)==0:               
            self.is_inference = True ### inference with src
        else:
            sys.stderr.write('error: bad number of input files {} {} {}\n'.format(len(fSRC),len(fTGT),len(LID)))
            sys.exit()

        ### check lid's exist in vocab
        for lid in LID:
            lid = self.vocab.beg_delim + lid + self.vocab.end_delim
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
                str_lid = self.vocab.beg_delim + LID[ifile] + self.vocab.end_delim
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
                if self.max_seq_size==0 and len(src)>self.maxtoksperline: 
                    sys.stderr.write('warning: src sentence sized of {} tokens truncated to {}\n'.format(len(src),self.maxtoksperline))
                    src = src[0:self.maxtoksperline]

                src.insert(0,self.vocab.str_bos)
                src.append(self.vocab.str_eos)
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
                    if self.max_seq_size==0 and len(tgt)>self.maxtoksperline: 
                        sys.stderr.write('warning: tgt sentence sized of {} tokens truncated to {}\n'.format(len(tgt),self.maxtoksperline))
                        tgt = tgt[0:self.maxtoksperline]

                    if self.is_inference: 
                        tgt.insert(0,self.vocab.str_bos)
                    else: ### is inference
                        tgt.insert(0,str_lid)
                    tgt.append(self.vocab.str_eos)
                ### tgt is: LID|<bos> my sentence <eos>
                #print("tgt: "+" ".join([str(e) for e in tgt]))

                if self.is_inference or config.max_seq_size==0 or (len(src)-2<=config.max_seq_size and len(tgt)-2<=config.max_seq_size):
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
        prev_tgt = []
        n_divergent = 0
        for slen, src, tgt in self.data: # src='<bos> my sentence <eos>' tgt='LID|<bos> my stencence <eos>'
            if self.is_align:
                tgt[0] = self.vocab.str_bos #i dont want LID as first token
                if self.is_inference or len(prev_tgt)==0 or np.random.random_sample()>=0.5: #### parallel (non divergent) example
                    isrc, iref_src, nunk_src = self.wrd2iwrd_ref(src, False)
                    itgt, iref_tgt, nunk_tgt = self.wrd2iwrd_ref(tgt, False) 
                else: #### divergent example
                    tgt=prev_tgt
                    n_divergent += 1
                    isrc, iref_src, nunk_src = self.wrd2iwrd_ref(src, True)
                    itgt, iref_tgt, nunk_tgt = self.wrd2iwrd_ref(tgt, True) 
                prev_tgt = tgt
            else:
                isrc, iref_src, nunk_src = self.src2isrc_iref(src)
                itgt, iref_tgt, nunk_tgt = self.tgt2itgt_iref(tgt)
#            print("src\t{}".format(" ".join(str(e) for e in src))) #src <bos> Saturday , April 26 will be a workday , and in exchange , Friday , May 2 will be a day off . <eos>
#            print("isrc\t{}".format(" ".join(str(e) for e in isrc))) #('isrc', [2, 11294, 16, 1413, 444, 53518, 17213, 13781, 0, 16, 15391, 31567, 27050, 16, 5325, 16, 8281, 340, 53518, 17213, 13781, 22502, 38251, 18, 3])
#            print("iref_src\t{}".format(" ".join(str(e) for e in iref_src))) #('iref', [7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18, 3])
#            print("nunk_src",nunk_src) #('nunk_src', 1)
            ### when not is_bitext itgt, iref are [], []
            ### otherwise
#            print("tgt\t{}".format(" ".join(str(e) for e in tgt))) #tgt LIDisFrench Le Samedi 25 avril sera un jour de travail pour avoir un jour de repos le vendredi 2 mai . <eos>
#            print("itgt\t{}".format(" ".join(str(e) for e in itgt))) #('itgt', [4, 7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18])
#            print("iref_tgt\t{}".format(" ".join(str(e) for e in iref_tgt))) #('iref', [7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18, 3])
#            print("nunk_tgt",nunk_tgt) #('nunk_tgt', 1)
            #### when is_inference
            #tgt is: <bos> Le Samedi 25 avril sera un jour de travail pour avoir un jour de repos le vendredi 2 mai . <eos>
            #itgt is: #('itgt', [2, 7554, 0, 435, 16826, 46749, 51581, 33513, 22509, 51097, 41074, 16816, 51581, 33513, 22509, 44419, 34269, 52654, 340, 35133, 18])
            batch.append([src, tgt, isrc, itgt, iref_src, iref_tgt, nunk_src, nunk_tgt])
            if len(batch)==config.batch_size:
                self.data_batch.append(batch)
                batch = []
        if len(batch): self.data_batch.append(batch) ### last batch
        self.data = []

        sys.stderr.write('(dataset contains {} batches with up to {} examples each. {} divergent examples)\n'.format(len(self.data_batch), config.batch_size, n_divergent))

    def wrd2iwrd_ref(self, wrd, is_divergent):
        iwrd = []
        ref = []
        nunk = 0
        for w in wrd:
            iwrd.append(self.vocab.get(w))
            if is_divergent: ref.append(1.0)
            else: ref.append(-1.0) 
            if iwrd[-1] == self.vocab.idx_unk: 
                nunk += 1
        ### <bos> and <eos> are never divergent
        ref[0] = -1.0 ### mark non-divergent (is aligned to any word in the other side)
        ref[-1] = -1.0
        return iwrd, ref, nunk

    def src2isrc_iref(self, src):
        isrc = []
        iref = []
        nunk = 0
        for s in src:
            isrc.append(self.vocab.get(s))
            if isrc[-1] == self.vocab.idx_unk: 
                nunk += 1
#            print(s,isrc[-1])
#        print("isrc:{}:".format(len(isrc)), isrc)
        return isrc, iref, nunk

    def tgt2itgt_iref(self, tgt): 
        itgt = []
        iref = []
        nunk = 0
        if self.is_bitext:
            for i,t in enumerate(tgt): 
                idx_t = self.vocab.get(t)
                if idx_t == self.vocab.idx_unk: 
                    nunk += 1
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
        SRC, TGT, REF_SRC, REF_TGT, RAW_SRC, RAW_TGT, NSRC_UNK, NTGT_UNK, LEN_SRC, LEN_TGT = [], [], [], [], [], [], [], [], [], []
        max_src, max_tgt = 0, 0
        for src, tgt, isrc, itgt, iref_src, iref_tgt, nsrc_unk, ntgt_unk in self.data_batch[index]:
            self.nsrc_tok += max(0,len(src)-2)
            self.ntgt_tok += max(0,len(tgt)-2)
            self.nsrc_unk += nsrc_unk
            self.ntgt_unk += ntgt_unk
            #### update max lenghts
            if len(isrc) > max_src: max_src = len(isrc)
            if len(itgt) > max_tgt: max_tgt = len(itgt)
            RAW_SRC.append(src)
            RAW_TGT.append(tgt)
            SRC.append(isrc)
            TGT.append(itgt)
            REF_SRC.append(iref_src)
            REF_TGT.append(iref_tgt)
            LEN_SRC.append(len(isrc)) ### '<bos> my sentence <eos>'
            LEN_TGT.append(len(itgt)) ### both iref and itgt have the same length
        ### add padding 
        for i in range(len(SRC)):
            while len(SRC[i]) < max_src: SRC[i].append(self.vocab.idx_pad) #<pad>
            while len(TGT[i]) < max_tgt: TGT[i].append(self.vocab.idx_pad) #<pad>
            if self.is_align:
                while len(REF_SRC[i]) < max_src: REF_SRC[i].append(-1.0) #not divergent
                while len(REF_TGT[i]) < max_tgt: REF_TGT[i].append(-1.0)
            else:
                while len(REF_SRC[i]) < max_src: REF_SRC[i].append(self.vocab.idx_pad)
                while len(REF_TGT[i]) < max_tgt: REF_TGT[i].append(self.vocab.idx_pad)
#            print("TGT",TGT[i])
#            print("REF",REF[i])        
        #print("BATCH max_src={} max_tgt={}".format(max_src,max_tgt))
        #print("LEN_SRC: {}".format(LEN_SRC))
        #print("LEN_TGT: {}".format(LEN_TGT))
        #for i in range(len(SRC)):
        #    print("EXAMPLE {}/{}".format(i,len(SRC)))
        #    print("  RAW_SRC: {}".format(" ".join([e for e in RAW_SRC[i]])))
        #    print("  RAW_TGT: {}".format(" ".join([e for e in RAW_TGT[i]])))
        #    print("  SRC: {}".format(" ".join([str(e) for e in SRC[i]])))
        #    print("  TGT: {}".format(" ".join([str(e) for e in TGT[i]])))
        #    print("  REF: {}".format(" ".join([str(e) for e in REF[i]])))
        return SRC, TGT, REF_SRC, REF_TGT, RAW_SRC, RAW_TGT, LEN_SRC, LEN_TGT



