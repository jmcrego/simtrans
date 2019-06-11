# -*- coding: utf-8 -*-
import sys
import codecs
import argparse
import random
import time
import pickle
import yaml
from sets import Set
from collections import defaultdict

def str_time():
    return time.strftime("[%Y-%m-%d_%X]", time.localtime())


class Freq(object):

    def __init__(self, filename, fnoun, fverb, fadj):
        nounFreq = defaultdict(int)
        verbFreq = defaultdict(int)
        adjFreq = defaultdict(int)
        sys.stderr.write('{} Building Freq statistics from: {}\n'.format(str_time(),filename))
        nsent = 0
        for line in open(filename, 'r'):
            line = line.rstrip()            
            toks = line.split()
            if len(toks)==0:
                nsent += 1
                continue
            elif len(toks)>=4:
                wrd = toks[0]
                lem = toks[1]
                pos = toks[2]
                #prb = float(toks[3])
                if pos.startswith('NC'):
                    nounFreq[lem] += 1
                elif pos.startswith('VM'):
                    verbFreq[lem] += 1
                elif pos.startswith('A'):
                    adjFreq[lem] += 1
            else:
                sys.stderr.write('warning1: unparsed {} entry \'{}\'\n'.format(nsent,line))

        sys.stderr.write('Noun: {} words\n'.format(len(nounFreq)))
        self.NOUN2TAG = self.split_in_sets(nounFreq, fnoun)

        sys.stderr.write('Verb: {} words\n'.format(len(verbFreq)))
        self.VERB2TAG = self.split_in_sets(verbFreq, fverb)

        sys.stderr.write('Adj: {} words\n'.format(len(adjFreq)))
        self.ADJ2TAG  = self.split_in_sets(adjFreq, fadj)


    def split_in_sets(self, wordFreq, ranges):        
        if ranges is not None: 
            RANGES = map(int, ranges.split(':'))
        else: 
            RANGES = []

        N = defaultdict(int)
        WORD2TAG = {}
        for w,f in wordFreq.iteritems():
            done = False
            beg = 0
            for end in RANGES:
                if f>=beg and f<=end: 
                    WORD2TAG[w] = "{}-{}".format(beg,end)
                    N["{}-{}".format(beg,end)] += 1
                beg = end + 1
            if f>=beg: 
                WORD2TAG[w] = "{}-".format(beg)
                N["{}-".format(beg)] += 1

        for t,n in sorted(N.iteritems()):
            sys.stderr.write("\t{}: {}\n".format(t,n))

        return WORD2TAG

    def inference(self, filename):
        WRD = []
        LEM = []
        POS = []
        nsent = 0
        for line in open(filename, 'r'):
            line = line.rstrip()
            toks = line.split()
            if len(toks)==0: ### end of sentence
                print(' '.join(self.sideConstraints(WRD, LEM, POS)))
                WRD = []
                LEM = []
                POS = []
                nsent += 1
            elif len(toks)>=4:
                WRD.append(toks[0])
                LEM.append(toks[1])
                POS.append(toks[2])
            else:
                sys.stderr.write('warning2: unparsed {} entry \'{}\'\n'.format(nsent,line))

    def sideConstraints(self, WRD, LEM, POS):
        sideconstraints = []
        ### length
        sentlength = len(WRD)
        if sentlength < 10: sideconstraints.append('<length:XS>')
        elif sentlength >= 10 and sentlength < 20: sideconstraints.append('<length:S>')
        elif sentlength >= 20 and sentlength < 30: sideconstraints.append('<length:M>')
        elif sentlength >= 30 and sentlength < 40: sideconstraints.append('<length:L>')
        elif sentlength >= 40: sideconstraints.append('<length:XL>')
        ### frequencies
        tag_of_nouns = Set()
        tag_of_verbs = Set()
        tag_of_adjs = Set()
        mood_of_verbs = Set()
        tens_of_verbs = Set()
        pers_of_verbs = Set()
        numb_of_verbs = Set()
        type_of_det = Set()
        for i in range(len(WRD)):
            wrd = WRD[i]
            lem = LEM[i]
            pos = POS[i]
            tag = 'NF'
            if pos.startswith('NC'): 
                if lem in self.NOUN2TAG: tag = self.NOUN2TAG[lem]
                tag_of_nouns.add(tag)

            elif pos.startswith('VM'): 
                if lem in self.VERB2TAG: tag = self.VERB2TAG[lem]
                tag_of_verbs.add(tag)
                mood_of_verbs.add(pos[2]) #I:indicative;   S:subjunctive;   M:imperative;   P:participle;   G:gerund;   N:infinitive
                tens_of_verbs.add(pos[3]) #P:present;   I:imperfect;   F:future;   S:past;   C:conditional
                pers_of_verbs.add(pos[4]) #1:1;   2:2;   3:3
                numb_of_verbs.add(pos[5]) #S:singular;   P:plural

            elif pos.startswith('A'): 
                if lem in self.ADJ2TAG: tag = self.ADJ2TAG[lem] 
                tag_of_adjs.add(tag)

            elif pos.startswith('D'):
                type_of_det.add(pos[1]) #A:article;   D:demonstrative;   I:indefinite;   P:possessive;   T:interrogative;   E:exclamative

#            sys.stderr.write("\t{} {} {} {}\n".format(wrd, lem, pos, tag))

        if len(tag_of_nouns)==1: 
            sideconstraints.append('<fnoun:{}>'.format(tag_of_nouns.pop()))

        if len(tag_of_verbs)==1: 
            sideconstraints.append('<fverb:{}>'.format(tag_of_verbs.pop()))

        if len(tag_of_adjs)==1: 
            sideconstraints.append('<fadj:{}>'.format(tag_of_adjs.pop()))

        if len(mood_of_verbs)==1: 
            mood = mood_of_verbs.pop()
            if mood != '0': 
                sideconstraints.append('<vmood:{}>'.format(mood))

        if len(tens_of_verbs)==1: 
            tens = tens_of_verbs.pop()
            if tens != '0': 
                sideconstraints.append('<vtense:{}>'.format(tens))

        if len(pers_of_verbs)==1: 
            pers = pers_of_verbs.pop()
            if pers != '0': 
                sideconstraints.append('<vperson:{}>'.format(pers))

        if len(numb_of_verbs)==1: 
            numb = numb_of_verbs.pop()
            if numb != '0': 
                sideconstraints.append('<vnumber:{}>'.format(numb))

        if len(type_of_det)==1: 
            det = type_of_det.pop()
            sideconstraints.append('<det:{}>'.format(det))

        return sideconstraints

##############################################################
### MAIN #####################################################
##############################################################

if __name__ == '__main__':

    name = sys.argv.pop(0)
    usage = '''{}  -i FILE -m FILE
       -i     FILE : input file with morfosyntactic analysis
       -m     FILE : model file
'''.format(name)

    fi = None
    fm = None
    fnoun = None
    fverb = None
    fadj = None
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   tok=="-i" and len(sys.argv): fi = sys.argv.pop(0)
        elif tok=="-m" and len(sys.argv): fm = sys.argv.pop(0)
        elif tok=="-fnoun" and len(sys.argv): fnoun = sys.argv.pop(0)
        elif tok=="-fverb" and len(sys.argv): fverb = sys.argv.pop(0)
        elif tok=="-fadj" and len(sys.argv): fadj = sys.argv.pop(0)
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

    if fm is not None:
        with open(fm, 'rb') as f: 
            Freq  = pickle.load(f)
            Freq.inference(fi)
    else:
        freq = Freq(fi,fnoun,fverb,fadj)        
        with open(fi+'.Freq', 'wb') as f: 
            pickle.dump(freq, f)

    sys.stderr.write('{} End\n'.format(str_time()))

