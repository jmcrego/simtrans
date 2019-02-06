# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import io
import os
import sys
import json
from shutil import copyfile
from vocab import Vocab
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
   -tok              FILE : src/tgt (json) onmt tokenization options

   Network topology:
   -net_wrd        STRING : word src/tgt embeddings size Ex: 256-0.3 (embedding_size-dropout)
   -net_enc        STRING : encoder layers (comma-separated list) Ex: c-512-3-0.3,b-512-0.3,b-512-0.3
                            Each layer follows the next formats:
                                -Convolutional (c-fiters-kernel_size-dropout)
                                -Bi-LSTM       (b-hidden_size-dropout) 
                                -LSTM
                                -GRU
   -net_snt        STRING : src sentence embedding: last, mean, max
   -net_dec        STRING : decoder layers (comma-separated list) Ex: l-2048-0.2 (type-embedding_size-dropout)
   -net_opt        STRING : Optimization method: adam, adagrad, adadelta, sgd, rmsprop
   -net_lid        STRING : list of LID tags (comma-separated list) Ex: English,French,German

   Training/Optimization:
   -opt_lr          FLOAT : initial learning rate [0.0001]
   -opt_decay       FLOAT : learning rate decay value [0.96]
   -opt_minlr       FLOAT : do not decay if learning rate is lower than this [0.0]
   -clip            FLOAT : gradient clipping value (0.0 for no clipping) [0.0]
   -max_seq_size      INT : src sentences larger than this number of words are filtered out [50]
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
        self.max_seq_size = 50
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
        self.net_wrd = None
        self.net_enc = None
        self.net_snt = None
        self.net_dec = None
        self.net_opt = None
        self.net_lid = None
        #optimization
        self.opt_lr = 0.0001
        self.opt_decay = 0.96
        self.opt_minlr = 0.0
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
            elif (tok=="-max_seq_size" and len(argv)):   self.max_seq_size = int(argv.pop(0))
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
            elif (tok=="-net_wrd" and len(argv)):        self.net_wrd = argv.pop(0)
            elif (tok=="-net_enc" and len(argv)):        self.net_enc = argv.pop(0)
            elif (tok=="-net_snt" and len(argv)):        self.net_snt = argv.pop(0)
            elif (tok=="-net_dec" and len(argv)):        self.net_dec = argv.pop(0)
            elif (tok=="-net_opt" and len(argv)):        self.net_opt = argv.pop(0)
            elif (tok=="-net_lid" and len(argv)):        self.net_lid = argv.pop(0)
            #optimization
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
                self.token = build_tokenizer(tok_opt)
                print('built tokenizer')

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
            if self.tok is not None: 
                with open(self.tok) as jsonfile: 
                    tok_opt = json.load(jsonfile)
                    print(tok_opt)
                    ### replaces/creates vocab option in token
                    tok_opt["vocabulary"] = self.mdir + '/vocab'
                    ### if exists bpe_model_path option, copy model to mdir and replaces bpe_model_path in token
                    if 'bpe_model_path' in tok_opt:
                        copyfile(tok_opt['bpe_model_path'], self.mdir + "/bpe")
                        ### replace token file with new bpe_model_path 
                        tok_opt['bpe_model_path'] = self.mdir + '/bpe'
                        with open(self.mdir + '/token', 'w') as outfile: 
                            json.dump(tok_opt, outfile)
                    else: 
                        ### copy token file
                        copyfile(self.tok, self.mdir + "/token")
                    print(tok_opt)
            #read vocab and token
            self.read_vocab_token()
            #write topology file
            with open(self.mdir + "/topology", 'w') as f: 
                for opt, val in vars(self).items():
                    if opt.startswith("net"): 
                        f.write("{} {}\n".format(opt,val))
            sys.stderr.write("learning from scratch\n")
        return  

    def write_config(self):
        if not os.path.exists(self.mdir): 
            os.makedirs(self.mdir)
        file = self.mdir + "/epoch"+str(self.last_epoch)+".config"
        with open(file,"w") as f:
            for name, val in vars(self).items():
                if name=="usage" or name.startswith("vocab") or name.startswith("token"): continue
                f.write("{} {}\n".format(name,val))

    def read_topology(self):
        argv = []
        with open(self.mdir + "/topology", 'r') as f:
            for line in f:
                opt, val = line.split()
                argv.append('-'+opt)
                argv.append(val)
        return argv

