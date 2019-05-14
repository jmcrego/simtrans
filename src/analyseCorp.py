# -*- coding: utf-8 -*-
#import numpy as np
import io
import sys
import yaml
from collections import defaultdict
from tokenizer import build_tokenizer

#PYTHONWARNINGS=ignore::yaml.YAMLLoadWarning

class File():
    def __init__(self, file, trn=None, token=None):
        self.file = file
        self.w2freq = defaultdict(int) ### frequency of words in document
        self.sentences = []
        self.Words = 0
        self.Lines = 0
        self.Lmax = 0
        self.Lmin = 999

        with open(file) as f:
            for line in f:
                line = line.strip()
                self.sentences.append(line)
                if token is not None: 
                    toks, _ = token.tokenize(str(line))
                else: 
                    toks = line.split()
                self.Lines += 1
                self.Lmax = max(len(toks), self.Lmax)
                self.Lmin = min(len(toks), self.Lmin)
                for w in toks: 
                    self.w2freq[w] += 1
                    self.Words += 1

        self.Lmean = 1.0 * self.Words / self.Lines
        self.Vocab = len(self.w2freq)

        TstTrn = ""
        if trn is not None:
            voc_oov = 0
            words_oov = 0
            sents_in_train = 0
            for w,n in self.w2freq.iteritems():
                if w not in trn.w2freq:
                    voc_oov += 1
                    words_oov += n
            for s in self.sentences:
                if s in trn.sentences:
                    sents_in_train += 1
            TstTrn = "\n\tOOV_voc={}\n\tOOV_words={}\n\tsents_in_train={}".format(voc_oov, words_oov, sents_in_train)

        sys.stderr.write('Read {}\n\tLines={}\n\tWords={}\n\tVocab={}\n\tLmean={:.2f}\n\tLmin={}\n\tLmax={}{}\n'.format(self.file, self.Lines, self.Words, self.Vocab, self.Lmean, self.Lmin, self.Lmax, TstTrn))

#    def words(self):
#        return self.w2freq.keys()


def main():

    name = sys.argv.pop(0)
    usage = '''{} -tok FILE -trn_src FILE -trn_tgt FILE -tst_src FILE -tst_tgt FILE
       -tok     FILE : options for tokenizer
       -trn_src FILE : train src file
       -trn_tgt FILE : train tgt file
       -tst_src FILE : test src file
       -tst_tgt FILE : test tgt file
'''.format(name)

    ftok = None
    ftrn_src= None
    ftrn_tgt= None
    ftst_src= None
    ftst_tgt= None
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   (tok=="-tok" and len(sys.argv)): ftok = sys.argv.pop(0)
        elif (tok=="-trn_src" and len(sys.argv)): ftrn_src = sys.argv.pop(0)
        elif (tok=="-trn_tgt" and len(sys.argv)): ftrn_tgt = sys.argv.pop(0)
        elif (tok=="-tst_src" and len(sys.argv)): ftst_src = sys.argv.pop(0)
        elif (tok=="-tst_tgt" and len(sys.argv)): ftst_tgt = sys.argv.pop(0)
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
            opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
            token = build_tokenizer(opts)

    if ftrn_src is not None:
        trn_src = File(ftrn_src,None,token)
        if ftst_src is not None:
            tst_src = File(ftst_src,trn_src,token)

    if ftrn_tgt is not None:
        trn_tgt = File(ftrn_tgt,None,token)
        if ftst_tgt is not None:
            tst_tgt = File(ftst_tgt,trn_tgt,token)

    sys.stderr.write('Done\n')

if __name__ == "__main__":
    main()



