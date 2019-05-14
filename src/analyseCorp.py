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
    usage = '''{}  -trn FILE [-tst FILE] [-tok FILE]
       -tok FILE : options for tokenizer
       -trn FILE : train file
       -tst FILE : test file
'''.format(name)

    ftok = None
    ftrn = None
    ftst = None
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   (tok=="-tok" and len(sys.argv)): ftok = sys.argv.pop(0)
        elif (tok=="-trn" and len(sys.argv)): ftrn = sys.argv.pop(0)
        elif (tok=="-tst" and len(sys.argv)): ftst = sys.argv.pop(0)
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

    if ftrn is not None:
        trn = File(ftrn,None,token)
        if ftst is not None:
            tst = File(ftst,trn,token)

    sys.stderr.write('Done\n')

if __name__ == "__main__":
    main()



