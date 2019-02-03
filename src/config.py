# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import io
import os
import sys
import json
from shutil import copyfile
from dataset import Vocab #, Embeddings
from tokenizer import build_tokenizer

class Config():

    def __init__(self, argv):
        self.usage="""usage: {}
*  -mdir              DIR : directory to save/restore models
   -batch_size        INT : number of examples per batch [32]
   -seed              INT : seed for randomness [12345]
   -h                     : this message

 [LEARNING OPTIONS]
*  -src_trn          FILE : src training files (comma-separated)
*  -tgt_trn          FILE : tgt training files (comma-separated)
*  -tgt_trn_lid      FILE : lid of tgt training files (comma-separated)

*  -src_val          FILE : src validation files (comma-separated)
*  -tgt_val          FILE : tgt validation files (comma-separated)
*  -tgt_val_lid      FILE : lid of tgt validation files (comma-separated)

   -voc              FILE : src/tgt vocab (needed to initialize learning)
   -tok              FILE : src/tgt json onmt tokenization options 

   -seq_size          INT : src sentences larger than this number of words are filtered out [50]

   Network topology:
   -net_wrd_len       INT : word src/tgt embeddings size [320]
   -net_conv_lens  STRING : kernel:units of each conv layer [3:1024,5:1024] (2 layers with 1024 cells and kernel sizes 3 and 5 respectively) []
   -net_blstm_lens STRING : units of src bi-lstm layers [1024,1024,1024] (3 layers with 1024 cells for each direction)
   -net_sentence   STRING : how src sentence embedding is formed from previous layer: last, mean, max [max]
   -net_lstm_len      INT : units of the tgt lstm layer [2048]

   -net_opt        STRING : Optimization method: adam, adagrad, adadelta, sgd, rmsprop [sgd]
   -net_lid        STRING : vocabulary of LID tags to be included in tgt_voc [] (Ex: LIDisEnglish,LIDisFrench,LIDisGerman)

   -dropout         FLOAT : dropout ratio applided to different layers [0.3]
   -opt_lr          FLOAT : initial learning rate [1.0]                              (use 0.0002 for adam)
   -opt_decay       FLOAT : learning rate decay value when opt_method='sgd' [0.9]    (use 0.98 for adam)
   -opt_minlr       FLOAT : do not decay if learning rate is lower than this [0.005] (use 0.0 for adam)
   -clip            FLOAT : gradient clipping value (0.0 for no clipping) [0.0]
   -max_sents         INT : Consider this number of sentences per epoch (0 for all) [0]
   -n_epochs          INT : train for this number of epochs [1]
   -reports           INT : report every this many batches [100]

 [INFERENCE OPTIONS]
*  -epoch             INT : epoch to use ([mdir]/epoch[epoch] must exist)
*  -src_tst          FILE : src testing file
   -tgt_tst          FILE : tgt testing file

   -show_sim              : output similarity score (target sentence is also passed through the encoder)
   -show_oov              : output number of OOVs in src/tgt OOVs 
   -show_emb              : output src/tgt sentence embeddings
   -show_snt              : output original src/tgt sentences
   -show_idx              : output idx of src/tgt sentences

+ Options marked with * must be set. The rest have default values.
+ If -mdir exists in learning mode, learning continues after restoring the last model
+ Training data is always shuffled at every epoch
+ -show_sim, -show_oov, -show_emb, -show_snt -show_idx can be used at the same time
""".format(sys.argv.pop(0))

        self.mdir = None
        self.seq_size = 50
        self.batch_size = 32
        self.seed = 12345
        #files
        self.src_trn = []
        self.tgt_trn = []
        self.tgt_trn_lid = []
        self.src_val = []
        self.tgt_val_lid = []
        self.voc = None
        self.tok = None
        #will be created
        self.vocab = None #vocabulary
#        self.embed = None #embedding
        self.token = None #onmt tokenizer
        #network
        self.net_wrd_len = 320
        self.net_blstm_lens = []
        self.net_conv_lens = []
        self.net_sentence = 'max'
        self.net_lstm_len = 2048
        self.net_opt = 'sgd'
        self.net_lid = []
        #optimization
        self.dropout = 0.3
        self.opt_lr = 1.0
        self.opt_decay = 0.9
        self.opt_minlr = 0.005
        self.clip = 0.0
        self.max_sents = 0
        self.n_epochs = 1
        self.last_epoch = 0 # epochs already run
        self.reports = 100
        #inference
        self.epoch = None
        self.src_tst = None
        self.tgt_tst = None
        self.show_sim = False
        self.show_oov = False
        self.show_emb = False
        self.show_snt = False
        self.show_idx = False

        self.parse(sys.argv)
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        if not self.mdir:
            sys.stderr.write("error: Missing -mdir option\n{}".format(self.usage))
            sys.exit()
        if len(self.mdir)>1 and self.mdir[-1]=="/": self.mdir = self.mdir[0:-1] ### delete ending '/'
        if self.src_tst: self.inference()
        elif len(self.src_trn) and len(self.tgt_trn) and len(self.tgt_trn_lid) and len(self.src_val) and len(self.tgt_val) and len(self.tgt_val_lid): self.learn()
        return

    def parse(self, argv):
        while len(argv):
            tok = argv.pop(0)
            if (tok=="-mdir" and len(argv)):             self.mdir = argv.pop(0)
            elif (tok=="-seq_size" and len(argv)):       self.seq_size = int(argv.pop(0))
            elif (tok=="-batch_size" and len(argv)):     self.batch_size = int(argv.pop(0))
            elif (tok=="-seed" and len(argv)):           self.seed = int(argv.pop(0))
            #files
            elif (tok=="-src_trn" and len(argv)):        self.src_trn = argv.pop(0).split(',')
            elif (tok=="-tgt_trn" and len(argv)):        self.tgt_trn = argv.pop(0).split(',')
            elif (tok=="-tgt_trn_lid" and len(argv)):    self.tgt_trn_lid = argv.pop(0).split(',')
            elif (tok=="-src_val" and len(argv)):        self.src_val = argv.pop(0).split(',')
            elif (tok=="-tgt_val" and len(argv)):        self.tgt_val = argv.pop(0).split(',')
            elif (tok=="-tgt_val_lid" and len(argv)):    self.tgt_val_lid = argv.pop(0).split(',')
            elif (tok=="-voc" and len(argv)):            self.voc = argv.pop(0)
            elif (tok=="-tok" and len(argv)):            self.tok = argv.pop(0)
            #network
            elif (tok=="-net_wrd_len" and len(argv)):    self.net_wrd_len = int(argv.pop(0))
            elif (tok=="-net_blstm_lens" and len(argv)): self.net_blstm_lens = map(int, argv.pop(0).split(','))
            elif (tok=="-net_conv_lens" and len(argv)):  self.net_conv_lens = argv.pop(0).split(',')
            elif (tok=="-net_sentence" and len(argv)):   self.net_sentence = argv.pop(0)
            elif (tok=="-net_lstm_len" and len(argv)):   self.net_lstm_len = int(argv.pop(0))
            elif (tok=="-net_opt" and len(argv)):        self.net_opt = argv.pop(0)
            elif (tok=="-net_lid" and len(argv)):        self.net_lid = argv.pop(0).split(',')
            #optimization
            elif (tok=="-dropout" and len(argv)):        self.dropout = float(argv.pop(0))
            elif (tok=="-opt_lr" and len(argv)):         self.opt_lr = float(argv.pop(0))
            elif (tok=="-opt_decay" and len(argv)):      self.opt_decay = float(argv.pop(0))
            elif (tok=="-opt_minlr" and len(argv)):      self.opt_minlr = float(argv.pop(0))
            elif (tok=="-clip" and len(argv)):           self.clip = float(argv.pop(0))
            elif (tok=="-max_sents" and len(argv)):      self.max_sents = int(argv.pop(0))
            elif (tok=="-n_epochs" and len(argv)):       self.n_epochs = int(argv.pop(0))
            elif (tok=="-reports" and len(argv)):        self.reports = int(argv.pop(0))
            #inference
            elif (tok=="-epoch" and len(argv)):          self.epoch = argv.pop(0)
            elif (tok=="-src_tst" and len(argv)):        self.src_tst = argv.pop(0)
            elif (tok=="-tgt_tst" and len(argv)):        self.tgt_tst = argv.pop(0)
            elif (tok=="-show_sim"):                     self.show_sim = True
            elif (tok=="-show_oov"):                     self.show_oov = True
            elif (tok=="-show_emb"):                     self.show_emb = True
            elif (tok=="-show_snt"):                     self.show_snt = True
            elif (tok=="-show_idx"):                     self.show_idx = True
            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

    def read_vocab_token(self):
        ### read vocabulary
        self.vocab = Vocab(self.mdir + "/vocab", self.net_lid)
        ### read tokenizer
        if os.path.exists(self.mdir + '/token'): 
            self.tok = self.mdir + '/token'
            with open(self.mdir + '/token') as jsonfile: 
                tok_opt = json.load(jsonfile)
                tok_opt["vocabulary"] = self.mdir + '/vocab'
                self.token = build_tokenizer(tok_opt)

    def inference(self):
        if not self.epoch:
            sys.stderr.write("error: Missing -epoch option\n{}".format(self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/epoch' + self.epoch + '.index'):
            sys.stderr.write('error: -epoch file {} cannot be find\n{}'.format(self.mdir + '/epoch' + self.epoch + '.index',self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/topology'): 
            sys.stderr.write('error: topology file: {} cannot be find\n{}'.format(self.mdir + '/topology',self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/vocab'): 
            sys.stderr.write('error: vocab file: {} cannot be find\n{}'.format(self.mdir + '/vocab',self.usage))
            sys.exit()
        argv = self.read_topology()
        self.parse(argv)
        self.dropout = 0.0
        #read vocab and token
        self.read_vocab_token()
        return  

    def learn(self):
        ###
        ### continuation
        ###
        if os.path.exists(self.mdir): 
            if not os.path.exists(self.mdir + '/topology'): 
                sys.stderr.write('error: topology file: {} cannot be find\n{}'.format(self.mdir + '/topology',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab'): 
                sys.stderr.write('error: vocab file: {} cannot be find\n{}'.format(self.mdir + '/vocab',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/checkpoint'): 
                sys.stderr.write('error: checkpoint file: {} cannot be find\ndelete dir {} ???\n{}'.format(self.mdir + '/checkpoint', self.mdir,self.usage))
                sys.exit()
            ### options in topology file override those passed in command line
            argv = self.read_topology()
            self.parse(argv)
            #read vocab and token
            self.read_vocab_token()
            ### update last epoch
            for e in range(999,0,-1):
                if os.path.exists(self.mdir+"/epoch{}.index".format(e)): 
                    self.last_epoch = e
                    break
            sys.stderr.write("learning continuation: last epoch is {}\n".format(self.last_epoch))
        ###
        ### learning from scratch
        ###
        else:
            if not os.path.exists(self.mdir): os.makedirs(self.mdir)
            #copy vocabularies
            copyfile(self.voc, self.mdir + "/vocab")
            #copy tokenizers if exist
            if self.tok: copyfile(self.tok, self.mdir + "/token")
            #read vocab and token
            self.read_vocab_token()
            #create embeddings
#            self.embed = Embeddings(self.vocab,self.net_wrd_len)
            #write topology file
            with open(self.mdir + "/topology", 'w') as f: 
                for opt, val in vars(self).items():
                    if not opt.startswith("net"): continue
                    if opt=="net_blstm_lens" or opt=="net_lid":
                        if len(val)>0: 
                            sval = ",".join([str(v) for v in val])
                            f.write("{} {}\n".format(opt,sval))
                    else:
                        f.write("{} {}\n".format(opt,val))
            sys.stderr.write("learning from scratch\n")
        return  

    def write_config(self):
        if not os.path.exists(self.mdir): 
            os.makedirs(self.mdir)
        file = self.mdir + "/epoch"+str(self.last_epoch)+".config"
        with open(file,"w") as f:
            for name, val in vars(self).items():
                if name=="usage" or name.startswith("embed") or name.startswith("vocab") or name.startswith("token"): continue
                f.write("{} {}\n".format(name,val))

    def read_topology(self):
        argv = []
        with open(self.mdir + "/topology", 'r') as f:
            for line in f:
                opt, val = line.split()
                argv.append('-'+opt)
                argv.append(val)
        return argv

