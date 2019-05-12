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
    def __init__(self, file, token=None):
        self.N = 0 ### N words in document
        self.Tf = defaultdict(int)
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
                    if w=='': 
                        #sys.stderr.write('warning: empty word >{}< in sentence >{}<\n'.format(w,line))
                        continue
                    self.Tf[w] += 1
                    self.N += 1
        sys.stderr.write('Read {} with {} sentences voc={}\n'.format(file,nsents,len(self.Tf)))

    def exists(self, w):
        return w in self.Tf

    def tf(self, w):
        return self.Tf[w] / (1.0 * self.N) ### returns 0.0 if does not exist

class TfIdf():

    def __init__(self):
        self.Tags = []  #[D]
        self.Vocab = [] #[V]
        self.TfIdf = [] #[V, D]
        self.Idf = [] #[V]
#        self.Tf = [] #[V, D]

    def learn(self, filetags, fmod, max, token=None):
        Vocab2Freq = defaultdict(int)
        Docs = []
        for filetag in filetags:
            (file,tag) = filetag.split(':')
            self.Tags.append(tag)
            Docs.append(Doc(file,token))
            for w,n in Docs[-1].Tf.iteritems():
                Vocab2Freq[w] += n

        for i,w_n in enumerate(sorted(Vocab2Freq.items(), key=lambda x: x[1], reverse=True)):
            if max>0 and i==max: break
            self.Vocab.append(w_n[0])

        D = len(Docs) ### number of documents
        for w in self.Vocab:
            N = sum(d.exists(w) for d in Docs) ### number of documents where appears w
            if N==0: N = D
            idf = math.log(D/1.0*N)
#            tf = []
            tfidf = []
            for doc in Docs:
#                tf.append(doc.tf(w))
                tfidf.append(doc.tf(w) * idf)
            self.TfIdf.append(tfidf)
#            self.Tf.append(tf)
            self.Idf.append(idf)

        self.TfIdf = np.asarray(self.TfIdf)
        self.Idf = np.asarray(self.Idf)
        ### normalize
        for d in range(self.TfIdf.shape[1]):
            vdoc = self.TfIdf[:,d]
            norm_vdoc = np.linalg.norm(vdoc)
            sys.stderr.write('Norm vdoc[{}]={}\n'.format(d,norm_vdoc))
            self.TfIdf[:,d] = vdoc / norm_vdoc

    def compute_distances(self,word2freq):
        ### build tst vector
        tf_tst = [0.0] * len(self.Vocab)
        for i,word in enumerate(self.Vocab):
            if word in word2freq:
                tf_tst[i] = word2freq[word]

        tfidf_tst = np.asarray(tf_tst) * self.Idf
        norm_tfidf_tst = np.linalg.norm(tfidf_tst)
        if norm_tfidf_tst == 0.0:
            print('norm:{}'.format(0.0))
            return
        tfidf_tst = tfidf_tst / norm_tfidf_tst
#        print('tfidf_tst: '+' '.join(["{}:{:.3f}".format(self.Vocab[i],e) for i,e in enumerate(tfidf_tst)]))

        res = {}
        for i,tag in enumerate(self.Tags):
            ### build doc_i vector
            vdoc = self.TfIdf[:,i]
            res[tag] = np.sum(tfidf_tst * vdoc)
#            print('vdoc['+ tag +']: '+' '.join(["{}:{:.3f}".format(self.Vocab[i],e) for i,e in enumerate(vdoc)])+' => '+str(res[tag]))

        out = []
        for r in sorted(res.items(), key=lambda x: x[1], reverse=True):
            out.append('{}:{}'.format(r[0],r[1]))
        print(' '.join(out))

    def inference(self, ftst, snt, token):
        word2freq = defaultdict(int)
        with open(ftst) as f:
            for line in f:
                line = line.strip()
                if token is not None: 
                    toks, _ = token.tokenize(str(line))
                else: 
                    toks = line.split(' ')
                for w in toks:
                    if w=='': 
                        #sys.stderr.write('warning: empty word >{}< in sentence >{}<\n'.format(w,line))
                        continue
                    word2freq[w] += 1
                if snt:
                    self.compute_distances(word2freq)
                    word2freq = defaultdict(int)

        if not snt:
            self.compute_distances(word2freq)

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



