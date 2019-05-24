import sys
import codecs
import argparse
import random
import edit_distance
import time
import pickle
import yaml
from bisect import bisect
from tokenizer import build_tokenizer
from collections import defaultdict

def str_time():
    return time.strftime("[%Y-%m-%d_%X]", time.localtime())

class SuffixArray(object):
    def __init__(self, filename, token):
        sys.stderr.write('{} Building SuffixArray from: {}\n'.format(str_time(),filename))
        sys.stderr.write('{} 1st pass to build vocab '.format(str_time()))
        self.vocab = {}
        self.vocab_inv = []
        self.EOS="<eos>"
        self.UNK="<unk>"
        self.vocab[self.EOS] = len(self.vocab)
        self.vocab_inv.append(self.EOS)
        self.EOS_id=0
        self.vocab[self.UNK] = len(self.vocab)
        self.vocab_inv.append(self.UNK)
        self.UNK_id=1
        for line in open(filename, 'r'):
            if token is not None: 
                toks, _ = token.tokenize(str(line))
            else: 
                toks = line.split()
            for word in toks:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    self.vocab_inv.append(word)
        sys.stderr.write('[{} distinct words]\n'.format(len(self.vocab_inv)))

        sys.stderr.write('{} 2nd pass to build suffix array '.format(str_time()))
        self.corpus_size = 0
        self.sentence_count = 0
        self.sentences = [0]
        self.len_sentences = []
        self.corpus = []
        nline = 0
        for line in open(filename, 'r'):
            nline += 1
            if nline % 10000 == 0:
                if nline % 100000 == 0: sys.stderr.write(str(nline))
                else: sys.stderr.write(".")
            if token is not None: 
                toks, _ = token.tokenize(str(line))
            else: 
                toks = line.split()
            word_ids = [self.vocab.get(w, self.UNK_id) for w in toks]
            word_ids.append(self.EOS_id)
            self.corpus.extend(word_ids)
            self.corpus_size += len(word_ids)
            self.sentences.append(self.corpus_size)
            self.len_sentences.append(len(word_ids)-1)
            self.sentence_count += 1
        self.suffixes = [0] * self.corpus_size
        self.lcp = [0] * self.corpus_size
        self.sentenceIds = [0] * self.corpus_size
        for i in range(self.corpus_size):
            self.suffixes[i] = i
            self.lcp[i] = 0
        sys.stderr.write('[{} lines]\n'.format(nline))
        sys.stderr.write('{} Sorting...\n'.format(str_time()))
        self.sort()
        sys.stderr.write('{} Computing LCP...\n'.format(str_time()))
        self.computeLCP()
        sys.stderr.write('{} Computing SentIds...\n'.format(str_time()))
        self.computeSentIds()

    def sort(self):
        sys.setrecursionlimit(self.corpus_size * 2)
        self.qsort(self.suffixes, 0, self.corpus_size-1)

    def computeLCP(self):
        h = 0
        for index in range(1, self.corpus_size):
            i = self.suffixes[index]
            j = self.suffixes[index - 1]
            h = 0
            while self.corpus[i+h]!=0 and self.corpus[j+h]!=0 and self.corpus[i+h] == self.corpus[j+h]:
                h += 1
            self.lcp[index] = h

    def computeSentIds(self):
        for index in range(self.corpus_size):
            self.sentenceIds[index] = self.getSentenceId(self.suffixes[index])

    def getSentenceId(self, corpusIndex):
        if corpusIndex > self.corpus_size: return -1
        index = bisect(self.sentences, corpusIndex)
        return index - 1

    def qsort(self, arr, begin, end, maxCompLength=15):
        if end > begin:
            index = random.randrange(begin, end)
            pivot = arr[index]

            arr[index], arr[end] = arr[end], arr[index]
            index = begin
            for i in range(begin, end):
                if self.compareSuffixes(arr[i], pivot, maxCompLength) <= 0:
                    arr[index], arr[i] = arr[i], arr[index]
                    index += 1

            arr[index], arr[end] = arr[end], arr[index]

            self.qsort(arr, begin, index-1, maxCompLength)
            self.qsort(arr, index+1, end, maxCompLength)

    def compareSuffixes(self, index1, index2, maxCompLength):
        for i in range(maxCompLength):
            if index1 + i < self.corpus_size and index2 + i >= self.corpus_size:
                return 1
            if index1 + i >= self.corpus_size and index2 + i < self.corpus_size:
                return -1
            diff = self.corpus[index1+i] - self.corpus[index2+i]
            if diff != 0:
                return diff
        return 0

    def comparePhrase(self, corpus_index, phrase):
        for i in range(len(phrase)):
            if corpus_index + i >= self.corpus_size:
                return -1
            diff = self.corpus[corpus_index + i] - phrase[i]
            if diff != 0:
                return diff
        return 0

    def findThePhrase(self, phrase, lowerBound, upperBound):
        lowBound = lowerBound
        upBound = upperBound
        while lowBound <= upBound and lowBound >= 0 and upBound >= 0:
            mid = int((lowBound + upBound) / 2)
            start = self.suffixes[mid]
            diff = self.comparePhrase(start, phrase)
            if diff == 0: return mid
            if diff < 0: lowBound = mid + 1
            elif diff > 0: upBound = mid -1
            else: return mid
        return -1

    def getSentenceIds(self, phrase):
        sents = []
        lowerBound = 0
        upperBound = self.corpus_size
        firstAppear = self.findThePhrase(phrase, lowerBound, upperBound)
        if firstAppear < 0: return sents
        upBound = firstAppear
        lowBound = firstAppear
        index = firstAppear
        while index > 0 and self.lcp[index] >= len(phrase):
            index -= 1
            lowBound = index
        index = firstAppear+1
        while index < self.corpus_size and self.lcp[index] >= len(phrase):
            upBound = index
            index += 1
        
        for idx in range(lowBound, upBound+1):
            sentId = self.sentenceIds[idx]
            sents.append(sentId)
        return sents
    
    def getSuffix(self, index, leng=3):
        start = index
        end = min(self.corpus_size, index+leng)
        return ' '.join([self.vocab_inv[wid] for wid in self.corpus[start:end]])

    def convert(self, toks):
        if len(toks) == 0:
            return toks

        if type(toks[0]) == str: ### I want the idx
            words = [self.vocab.get(w, self.UNK_id) for w in toks]
        else: ### I want the strings
            words = [self.vocab_inv[w] for w in toks]
        return words

    def getSubPhrase(self, phrases, minNgramLength = 1, maxNgramLength = 4, remove_UNK=True):
        subphrase_list = []
        for phrase in phrases:
            maxNgram = maxNgramLength
            if maxNgram > len(phrase): maxNgram = len(phrase)
            for i in range(len(phrase)):
                for j in range(i+minNgramLength, min(i + maxNgram, len(phrase)) + 1):
                    sub_phrase = phrase[i:j]
                    if remove_UNK and self.UNK_id in sub_phrase: 
                        continue
                    if sub_phrase not in subphrase_list: 
                        subphrase_list.append(sub_phrase)
        return subphrase_list
  
    def query(self, ltoks, mingram, maxngram, nbest, idx_tst):
        if idx_tst >= 0: #single sentence
            print("[{}]\t{}".format(idx_tst,' '.join(ltoks[0])))

        query_idx = []
        for toks in ltoks:
            query_idx.append(self.convert(toks))

        subphrases = self.getSubPhrase(query_idx, mingram, maxngram)
        counts = defaultdict(int)
        for subphrase in subphrases:
            result = self.getSentenceIds(subphrase)
            for idx_trn in result:
                counts[idx_trn] += 1
        sorted_counts = sorted(counts.items(), key = lambda x:x[1], reverse=True)
        for idx_trn, ngrams_count in sorted_counts[:nbest]:
            entry = []
            entry.append("{}".format(ngrams_count)) ### ngrams_counts
            trn_vec_idx = self.corpus[self.sentences[idx_trn]:self.sentences[idx_trn+1]-1]
            if idx_tst >= 0: #single sentence
                sm = edit_distance.SequenceMatcher(a=query_idx[0], b=trn_vec_idx)
                entry.append("{:.4f}".format(sm.ratio())) ### edit distance
                entry.append("{}".format(idx_tst)) ### index tst
            entry.append("{}".format(idx_trn)) ### index trn
            entry.append(' '.join(self.convert(trn_vec_idx))) ### trn
            print('\t'.join(entry))
  
    def queryfile(self, filename, token, minngram, maxngram, nbest, testset):
        ntst = 0
        test_toks = []
        for line in open(filename, 'r'):
            if token is not None: toks, _ = token.tokenize(str(line))
            else: toks = line.split()
            if testset: test_toks.append(toks)
            else: output = self.query([toks], minngram, maxngram, nbest, ntst)
            ntst += 1
        output = self.query(test_toks, minngram, maxngram, nbest, -1)


if __name__ == '__main__':

    Nbest = 10
    minNgram = 2
    maxNgram = 4
    testSet = False
    name = sys.argv.pop(0)
    usage = '''{}  -mod FILE -trn FILE -tst FILE [-tok FILE] [-nbest INT]
       -mod     FILE : Suffix Array model file
       -trn     FILE : train file
       -tst     FILE : test file
       -tok     FILE : options for tokenizer
       -Nbest    INT : show [INT]-best similar sentences (default 10)
       -minNgram INT : min length for test ngrams (default 2)
       -maxNgram INT : max length for test ngrams (default 4)
       -testSet      : collect similar sentences to the entire test set rather than to each input sentence (default false)
'''.format(name)

    ftok = None
    fmod = None
    ftrn = None
    ftst = None
    testSet = False
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if   tok=="-tok" and len(sys.argv): ftok = sys.argv.pop(0)
        elif tok=="-mod" and len(sys.argv): fmod = sys.argv.pop(0)
        elif tok=="-trn" and len(sys.argv): ftrn = sys.argv.pop(0)
        elif tok=="-tst" and len(sys.argv): ftst = sys.argv.pop(0)
        elif tok=="-Nbest" and len(sys.argv): Nbest = int(sys.argv.pop(0))
        elif tok=="-minNgram" and len(sys.argv): minNgram = int(sys.argv.pop(0))
        elif tok=="-maxNgram" and len(sys.argv): maxNgram = int(sys.argv.pop(0))
        elif tok=="-testSet": testSet = True
        elif tok=="-h":
            sys.stderr.write("{}".format(usage))
            sys.exit()
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()

    if ftrn is None and ftst is None:
        sys.stderr.write('error: -trn and/or -tst options must be set\n')
        sys.stderr.write("{}".format(usage))
        sys.exit()

    sys.stderr.write('{} Start\n'.format(str_time()))
    token = None
    if ftok is not None:
        with open(ftok) as yamlfile: 
            opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
            token = build_tokenizer(opts)

    sa = None
    if ftrn is not None:
        sa = SuffixArray(ftrn, token)
        if fmod is not None:
            with open(fmod, 'wb') as f: pickle.dump(sa, f)

    if ftst is not None:
        if sa is None and fmod is not None:
            with open(fmod, 'rb') as f: sa  = pickle.load(f)
            sys.stderr.write('{} Read model from: {}\n'.format(str_time(),fmod))
            sa.queryfile(ftst,token,minNgram,maxNgram,Nbest,testSet)        
        
    sys.stderr.write('{} End\n'.format(str_time()))

