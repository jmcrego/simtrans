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
#from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')

class Vocab():

    def __init__(self, dict_file, net_lid=[]):

        self.tok_to_idx = {}
        self.idx_to_tok = []

        self.idx_unk = 0
        self.str_unk = "<unk>"
        self.idx_to_tok.append(self.str_unk)
        self.tok_to_idx[self.str_unk] = len(self.tok_to_idx) #0

        self.idx_pad = 1
        self.str_pad = "<pad>"
        self.idx_to_tok.append(self.str_pad)
        self.tok_to_idx[self.str_pad] = len(self.tok_to_idx) #1

        self.idx_bos = 2
        self.str_bos = "<bos>"
        self.idx_to_tok.append(self.str_bos)
        self.tok_to_idx[self.str_bos] = len(self.tok_to_idx) #2

        self.idx_eos = 3
        self.str_eos = "<eos>"
        self.idx_to_tok.append(self.str_eos)
        self.tok_to_idx[self.str_eos] = len(self.tok_to_idx) #3

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
                    sys.stderr.write('error: repeated key \'{}\' in vocab (check if LIDs exist in vocab) at line={}\n'.format(lid,nline))
                    sys.exit()
                self.idx_to_tok.append(line)
                self.tok_to_idx[line] = len(self.tok_to_idx)

        self.length = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries) {}\n'.format(self.length, dict_file))

    def __len__(self):
        return len(self.idx_to_tok)

#    def __iter__(self):
#        for tok in self.idx_to_tok:
#            yield tok

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
            return self.idx_unk
        return self.tok_to_idx[s]


