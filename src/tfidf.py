# -*- coding: utf-8 -*-
#import numpy as np
import io
import sys
import math
import os.path
import numpy as np
import pickle
import yaml
from collections import defaultdict
from tokenizer import build_tokenizer


class Doc():
    def __init__(self, words, file, token=None):
        self.N = 0 ### num words in document
        self.w2freq = defaultdict(int) ### frequency of words in document

        if len(words):
            for w in words:
                self.w2freq[w] += 1
                self.N += 1
            sys.stderr.write('Read {} words, voc={}\n'.format(self.N,len(self.w2freq)))

        elif len(file):
            nsents = 0
            with open(file) as f:
                for line in f:
                    nsents += 1
                    line = line.strip()
                    if token is not None: 
                        toks, _ = token.tokenize(str(line))
                    else: 
                        toks = line.split(' ')
                    for w in toks:
                        if w=='': continue
                        self.w2freq[w] += 1
                        self.N += 1
            sys.stderr.write('Read {} with {} sentences, {} words, voc={}\n'.format(file,nsents,self.N,len(self.w2freq)))

        ### compute Tf (freq / N) and norm of the resulting vector

    def exists(self, w):
        return w in self.w2freq

    def Tf(self, w):
        #returns 0.0 if w not in w2freq
        return self.w2freq[w] / float(self.N)

class TfIdf():

    def __init__(self):
        self.Tags = []  #[D]
        self.Vocab = [] #[V]
        self.TfIdf = [] #[V, D]
        self.Idf = [] #[V]
#        self.Tf = [] #[V, D]

    def learn(self, filetags, fmod, max, token=None):
        w2freq = defaultdict(int)
        Docs = []
        for filetag in filetags:
            (file,tag) = filetag.split(':')
            self.Tags.append(tag)
            Docs.append(Doc([],file,token))
            for w,n in Docs[-1].w2freq.iteritems():
                w2freq[w] += n

        ### computes self.Vocab as the [max] most frequent of all domains
        for i,w_n in enumerate(sorted(w2freq.items(), key=lambda x: x[1], reverse=True)):
            if max>0 and i==max: break
            self.Vocab.append(w_n[0])

        D = len(Docs) ### number of documents
        normD = np.zeros(D) ### norm of each vector 
        for w in self.Vocab:
            N = sum(d.exists(w) for d in Docs) ### number of documents where appears w
            idf = math.log(D/1.0*N)
            tfidf = []
            for d,doc in enumerate(Docs):
                val = doc.Tf(w) * idf
                tfidf.append(val)
                normD[d] += np.power(val,2.0)
            self.TfIdf.append(tfidf)
            self.Idf.append(idf)

        self.TfIdf = np.asarray(self.TfIdf)
        self.Idf = np.asarray(self.Idf)
        ### normalize
        #for d in range(D):
        #    vdoc = self.TfIdf[:,d]
        #    normD[d] = np.linalg.norm(vdoc)
        ### normalize
        normD = np.power(normD,0.5)
        for i in range(len(self.TfIdf)):
            self.TfIdf[i] = np.divide(self.TfIdf[i], normD)

    def compute_distances(self,words,ftst,token=None):
        doc = Doc(words,ftst,token)
        ### build tst tfidf vector
        tfidf_tst = []
        norm = 0.0
        for i,w in enumerate(self.Vocab):
            val = doc.Tf(w) * self.Idf[i]
            tfidf_tst.append(val)
            norm += np.power(val,2.0)
        norm = np.power(norm,0.5)
        if norm == 0.0:
            print('norm:0')
            return

        tfidf_tst = np.asarray(tfidf_tst)
        tfidf_tst = np.divide(tfidf_tst, norm)
        print('vtst: '+' '.join(["{}:{:.3f}".format(self.Vocab[i],e) for i,e in enumerate(tfidf_tst)]))

        res = {}
        for d,tag in enumerate(self.Tags):
            vdoc = self.TfIdf[:,d]
            res[tag] = np.sum(tfidf_tst * vdoc)
            print('vdoc['+ tag +']: '+' '.join(["{}:{:.3f}".format(self.Vocab[i],e) for i,e in enumerate(vdoc)])+' => '+str(res[tag]))

        out = []
        for r in sorted(res.items(), key=lambda x: x[1], reverse=True):
            out.append('{}:{}'.format(r[0],r[1]))
        print(' '.join(out))

    def inference(self, ftst, snt, token):

        if not snt:
            self.compute_distances([],ftst,token)
            return

        with open(ftst) as f:
            for line in f:
                line = line.strip()
                if token is not None: 
                    toks, _ = token.tokenize(str(line))
                else: 
                    toks = line.split(' ')
                self.compute_distances(toks,'',token)

    def debug(self):
        print(self.Tags)

        for i,v in enumerate(self.Vocab):
            print(i,v)

        for i,d in enumerate(self.TfIdf):
            print(i,d)



def main():

    name = sys.argv.pop(0)
    usage = '''{} -tok FILE -mod FILE ([-trn STRING]+ | -tst FILE [-snt])
       -tok   FILE : options for tokenizer
       -mod   FILE : tfidf model file (to create/save)
       -tst   FILE : file used for inference
       -trn STRING : file:tag used for the given domain
       -max      N : max vocabulary size (default 0: use all)
       -snt        : compute tfidf values for each sentence rather the entire tst file
'''.format(name)

    ftok = None
    fmod = None
    vtrn = []
    ftst = None
    max_voc_size = 0
    snt = False #### compute inference over whole test-set
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   (tok=="-tok" and len(sys.argv)): ftok = sys.argv.pop(0)
        elif (tok=="-mod" and len(sys.argv)): fmod = sys.argv.pop(0)
        elif (tok=="-trn" and len(sys.argv)): vtrn.append(sys.argv.pop(0))
        elif (tok=="-tst" and len(sys.argv)): ftst = sys.argv.pop(0)
        elif (tok=="-max" and len(sys.argv)): max_voc_size = int(sys.argv.pop(0))
        elif (tok=="-snt"): snt = True
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


    tfidf = TfIdf()
    #############################
    ### create/read the model ###
    #############################
    if len(vtrn):
        if os.path.exists(fmod):
            sys.stderr.write('error: the path {} already exists\n'.format(fmod))
            sys.exit()
        tfidf.learn(vtrn,fmod,max_voc_size,token)
        fout = open(fmod,'w')
        pickle.dump(tfidf, fout)
        fout.close()
        #tfidf.debug()
        sys.stderr.write('Wrote model (V, D) = {}\n'.format(tfidf.TfIdf.shape))
    else:
        fin = open(fmod,'r')
        tfidf = pickle.load(fin)
        fin.close()
        sys.stderr.write('Read model (V, D) = {}\n'.format(tfidf.TfIdf.shape))

    #################
    ### inference ###
    #################
    if ftst is not None:
        tfidf.inference(ftst,snt,token)

    sys.stderr.write('Done\n')

if __name__ == "__main__":
    main()



