# -*- coding: utf-8 -*-
#import numpy as np
import io
import sys
import math
import yaml
from collections import defaultdict
from tokenizer import build_tokenizer

class SentIdf(): #each sentence is considered a document

    def __init__(self, file=None):
        self.n_sents_containing_w = defaultdict(int)
        self.n_sents = 0

        if file is not None:
            with open(file) as f:
                nline = 0
                for line in f:
                    nline += 1
                    if nline == 1:
                        self.n_sents = int(line.rstrip('\n'))
                    else:
                        (w, n) = line.rstrip('\n').split(' ')
                        self.n_sents_containing_w[w] = int(n)


    def add(self, file, token=None):
        with open(file) as f:
            for line in f:
                ### split line
                line = line.strip('\n')
                if token is not None: 
                    toks, _ = token.tokenize(str(line))
                else: 
                    toks = line.split(' ')
                ### update n_sents_containing_w and n_sents
                self.n_sents += 1
                for w in set(toks):
                    self.n_sents_containing_w[w] += 1

    def save(self, file):
        with open(file,"w") as f:
            f.write('{}\n'.format(self.n_sents))
            for w, n in self.n_sents_containing_w.items(): 
                f.write("{} {}\n".format(w,n))

    def idf(self, w):
        idf = 0.0
        if w in self.n_sents_containing_w:
            idf = math.log(self.n_sents / (1.0+self.n_sents_containing_w[w]))
        return idf

    def tfidf(self, words, use_tf=False):
        if use_tf:
            tf = defaultdict(int)
            for w in words: tf[w] += 1

        tfidf = []
        for w in words:
            tf = 1.0
            if use_tf: 
                tf = tf[w]
            tfidf.append(tf*self.idf(w))
        return tfidf



def main():

    name = sys.argv.pop(0)
    usage = '''{} [-data FILE] ( -save FILE | -load FILE )
       -tok  FILE : options for tokenizer
       -data FILE : file used to learn/inference
       -save FILE : save tfidf model after building it with data file
       -load FILE : load tfidf model and use it for inference on data file
'''.format(name)

    ftok = None
    fsave = None
    fload = None
    fdata = []
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   (tok=="-tok" and len(sys.argv)):  ftok = sys.argv.pop(0)
        elif (tok=="-save" and len(sys.argv)): fsave = sys.argv.pop(0)
        elif (tok=="-load" and len(sys.argv)): fload = sys.argv.pop(0)
        elif (tok=="-data" and len(sys.argv)): fdata.append(sys.argv.pop(0))
        elif (tok=="-h"):
            sys.stderr.write("{}".format(usage))
            sys.exit()
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()

    token = None
    if ftok is not None:
        with open(ftok) as yamlfile: 
            opts = yaml.load(yamlfile)
            token = build_tokenizer(opts)

    if fsave is not None and len(fdata):
        sys.stderr.write('Learning mode\n')
        sentIdf = SentIdf()
        for f in fdata:
            sys.stderr.write('\treading {}\n'.format(f))
            sentIdf.add(f,token)
        sys.stderr.write('Model saved in {}\n'.format(fsave))
        sentIdf.save(fsave)

    if fload is not None and len(fdata):
        sys.stderr.write('Inference mode. Model in {}\n'.format(fload))
        sentIdf = SentIdf(fload)

        for file in fdata:
            with open(file) as f: 
                for line in f:
                    line = line.strip('\n')
                    if token is not None: 
                        toks, _ = token.tokenize(str(line))
                    else: 
                        toks = line.split(' ')

                    tfidf = sentIdf.tfidf(toks,use_tf=False)
                    sys.stdout.write(" ".join(toks)+'\n')
                    for i in range(len(toks)):
                        sys.stdout.write("{:.8f}\t{}\n".format(tfidf[i],toks[i]))


if __name__ == "__main__":
    main()






