# -*- coding: utf-8 -*-
import sys
import codecs
import argparse
import random
import time
import pickle
import yaml
import numpy as np
from sets import Set
from collections import defaultdict

def str_time():
    return time.strftime("[%Y-%m-%d_%X]", time.localtime())

    
class Model(object):

    def __init__(self, filename, setsnoun, setsverb, setsadj, frq_or_rnd):
        Nnoun = 0
        Nverb = 0
        Nadj = 0
        noun2n = defaultdict(int)
        verb2n = defaultdict(int)
        adj2n = defaultdict(int)

        sys.stderr.write('{} Building statistics from: {}\n'.format(str_time(),filename))
        nsent = 0
        for line in open(filename, 'r'):
            line = line.rstrip()            
            toks = line.split()
            if len(toks)==0:
                nsent += 1
                continue
            elif len(toks)>=4:
                wrd = toks[0]
                pos = toks[2]
                if pos.startswith('NC'):
                    noun2n[wrd] += 1
                    Nnoun += 1
                elif pos.startswith('VM'):
                    verb2n[wrd] += 1
                    Nverb += 1
                elif pos.startswith('A'):
                    adj2n[wrd] += 1
                    Nadj += 1
            else:
                sys.stderr.write('warning1: unparsed {} entry \'{}\'\n'.format(nsent,line))

        sys.stderr.write('Noun: V={}\n'.format(len(noun2n)))
        if frq_or_rnd: 
            self.NOUN2TAG = self.split_in_fsets(noun2n, Nnoun, setsnoun, filename, "fnoun")
        else: 
            self.NOUN2TAG = self.split_in_rsets(noun2n, Nnoun, setsnoun, filename, "rnoun")

        sys.stderr.write('Verb: V={}\n'.format(len(verb2n)))
        if frq_or_rnd:
            self.VERB2TAG = self.split_in_fsets(verb2n, Nverb, setsverb, filename, "fverb")
        else:
            self.VERB2TAG = self.split_in_rsets(verb2n, Nverb, setsverb, filename, "rverb")

        sys.stderr.write('Adj: V={}\n'.format(len(adj2n)))
        if frq_or_rnd:
            self.ADJ2TAG = self.split_in_fsets(adj2n, Nadj, setsadj, filename, "fadj")
        else:
            self.ADJ2TAG = self.split_in_rsets(adj2n, Nadj, setsadj, filename, "radj")

    def split_in_fsets(self, wordDict, nwords, nsets, ofile, mytag):
        drange = nwords / nsets
        total_sofar = 0
        N = defaultdict(int)
        M = defaultdict(int)
        WORD2TAG = {}
        maxim = drange
        k = ord('A')
        for w,f in sorted(wordDict.iteritems(), key=lambda (k,v): v): #from smaller to larger
            if maxim < total_sofar:
                maxim += drange
                k += 1
            WORD2TAG[w] = chr(k)
            N[chr(k)] += 1
            M[chr(k)] += f
            total_sofar += f

        of = open(ofile+"."+mytag, "w") 
        for w,t in sorted(WORD2TAG.iteritems(), key=lambda (k,v): v): #from smaller to larger (length)
            f = wordDict[w]
            of.write("{}\t{}\t{}\n".format(t,w,f))

        for t,n in sorted(N.iteritems()):
            m = M[t]
            sys.stderr.write("\t{} {}: type={} tok={}\n".format(mytag,t,n,m))

        return WORD2TAG


    def split_in_rsets(self, wordDict, nwords, nsets, ofile, mytag):
        V = np.full(shape=nsets, fill_value=0, dtype=np.int)
        total_sofar = 0
        N = defaultdict(int)
        M = defaultdict(int)
        WORD2TAG = {}
        ord_a = ord('A')
        for w,f in sorted(wordDict.iteritems(), key=lambda (k,v): v): #from smaller to larger (length)
            i_min = np.argmin(V)
            myclass = chr(ord_a+i_min)
            WORD2TAG[w] = myclass
            V[i_min] += f
            N[myclass] += 1
            M[myclass] += f
            total_sofar += f

        of = open(ofile+"."+mytag, "w") 
        for w,t in sorted(WORD2TAG.iteritems(), key=lambda (k,v): v): #from smaller to larger (length)
            f = wordDict[w]
            of.write("{}\t{}\t{}\n".format(t,w,f))

        for t,n in sorted(N.iteritems()):
            m = M[t]
            sys.stderr.write("\t{} {}: type={} tok={}\n".format(mytag,t,n,m))

        return WORD2TAG

    def inference(self, filename, p_any, str_any, ling):
        WRD = []
        POS = []
        nsent = 0
        for line in open(filename, 'r'):
            line = line.rstrip()
            toks = line.split()
            if len(toks)==0: ### end of sentence
                print(' '.join(self.sideConstraints(WRD, POS, p_any, str_any, ling)))
                WRD = []
                POS = []
                nsent += 1
            elif len(toks)>=4:
                WRD.append(toks[0])
                POS.append(toks[2])
            else:
                sys.stderr.write('warning2: unparsed {} entry \'{}\'\n'.format(nsent,line))

    def sideConstraints(self, WRD, POS, p_any, str_any, ling):
        sideconstraints = []
        ### clusters
        set_of_nouns = Set()
        set_of_verbs = Set()
        set_of_adjs = Set()
        ### pos
        mood_of_verbs = Set()
        tense_of_verbs = Set()
        n_nouns = 0
        n_verbs = 0
        n_adjs = 0
        for i in range(len(WRD)):
            wrd = WRD[i]
            pos = POS[i]
            if pos.startswith('NC'): ### noun common
                n_nouns += 1
                if wrd in self.NOUN2TAG: set_of_nouns.add(self.NOUN2TAG[wrd])
                
            elif pos.startswith('VM'): ### verb main
                n_verbs += 1
                if wrd in self.VERB2TAG: set_of_verbs.add(self.VERB2TAG[wrd])
                mood_of_verbs.add(pos[2]) #I:indicative;   S:subjunctive;   M:imperative;   P:participle;   G:gerund;   N:infinitive
                tense_of_verbs.add(pos[3]) #P:present;   I:imperfect;   F:future;   S:past;   C:conditional

            elif pos.startswith('A'): 
                n_adjs += 1
                if wrd in self.ADJ2TAG: set_of_adjs.add(self.ADJ2TAG[wrd])


        if len(set_of_nouns)==1 and random.random() >= p_any: sideconstraints.append('<noun:{}>'.format(set_of_nouns.pop()))
        else: sideconstraints.append('<noun:{}>'.format(str_any))

        if len(set_of_verbs)==1 and random.random() >= p_any: sideconstraints.append('<verb:{}>'.format(set_of_verbs.pop()))
        else: sideconstraints.append('<verb:{}>'.format(str_any))

        if len(set_of_adjs)==1 and random.random() >= p_any:  sideconstraints.append('<adj:{}>'.format(set_of_adjs.pop()))
        else: sideconstraints.append('<adj:{}>'.format(str_any))

        if ling:
            mood = '0'
            if len(mood_of_verbs)==1: mood = mood_of_verbs.pop()
            if mood != '0' and random.random() > p_any: sideconstraints.append('<mood:{}>'.format(mood))
            else: sideconstraints.append('<mood:{}>'.format(str_any))

            tense = '0'
            if len(tense_of_verbs)==1: tense = tense_of_verbs.pop()
            if tense != '0' and random.random() > p_any: sideconstraints.append('<tense:{}>'.format(tense))
            else: sideconstraints.append('<tense:{}>'.format(str_any))

        return sideconstraints

##############################################################
### MAIN #####################################################
##############################################################

if __name__ == '__main__':

    name = sys.argv.pop(0)
    usage = '''{}  -i FILE [-m FILE] [-snoun INT] [-sverb INT] [-sadj INT] [-rnd] [-p FLOAT] [-ling] [-seed INT]
       -i       FILE : input file with morphosyntactic analysis

   Learning Options
       -snoun    INT : number of noun sets (default 2)
       -sverb    INT : number of verb sets (default 2)
       -sadj     INT : number of adj sets (default 2)
       -rnd          : split words in random sets rather than frequency (default False)
       -seed     INT : seed for randomness (default 1234)

   Inference Options
       -m       FILE : model file
       -p_any  FLOAT : probability of generating using 'X' (any) for tags (default 0.3)
       -s_any STRING : use 'X' for tags meaning any tag (default X)
       -ling         : use verb mood and tense costraints (default False)

'''.format(name)

    fi = None
    fm = None
    snoun = 2
    sverb = 2
    sadj = 2
    ling = False
    frq_or_rnd = True
    p_any = 0.3
    str_any = 'X'
    seed = 1234
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   tok=="-i" and len(sys.argv): fi = sys.argv.pop(0)
        elif tok=="-m" and len(sys.argv): fm = sys.argv.pop(0)
        elif tok=="-snoun" and len(sys.argv): snoun = int(sys.argv.pop(0))
        elif tok=="-sverb" and len(sys.argv): sverb = int(sys.argv.pop(0))
        elif tok=="-sadj" and len(sys.argv): sadj = int(sys.argv.pop(0))
        elif tok=="-p_any" and len(sys.argv): p_any = float(sys.argv.pop(0))
        elif tok=="-s_any" and len(sys.argv): str_any = sys.argv.pop(0)
        elif tok=="-seed" and len(sys.argv): seed = int(sys.argv.pop(0))
        elif tok=="-rnd": frq_or_rnd = False
        elif tok=="-ling": ling = True
        elif tok=="-h":
            sys.stderr.write("{}".format(usage))
            sys.exit()
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()

    if fi is None:
        sys.stderr.write('error: -i option must be set\n')
        sys.stderr.write("{}".format(usage))
        sys.exit()

    sys.stderr.write('{} Start\n'.format(str_time()))

    random.seed(seed)
    if fm is not None:
        with open(fm, 'rb') as f: 
            Mod  = pickle.load(f)
            Mod.inference(fi,p_any,str_any,ling)
    else:
        Mod = Model(fi,snoun,sverb,sadj,frq_or_rnd)
        if frq_or_rnd:
            mod = 'Frq'
        else:
            mod = 'Rnd'
        with open(fi+'.'+mod, 'wb') as f: 
            pickle.dump(Mod, f)

    sys.stderr.write('{} End\n'.format(str_time()))

