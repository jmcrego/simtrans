# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import io
import os
import sys
import json
from shutil import copyfile
from dataset import Vocab, Embeddings
from tokenizer import build_tokenizer

class Config():

    def __init__(self, argv):
        self.usage="""usage: {}
*  -mdir              DIR : directory to save/restore models
   -seq_size          INT : sentences larger than this number of src/tgt words are filtered out [50]
   -batch_size        INT : number of examples per batch [32]
   -seed              INT : seed for randomness [12345]
   -h                     : this message

 [LEARNING OPTIONS]
*  -src_trn          FILE : src training data
*  -tgt_trn          FILE : tgt training data
*  -lid_trn          FILE : lid training data

*  -src_val          FILE : src validation data
*  -tgt_val          FILE : tgt validation data
*  -lid_val          FILE : lid validation data

   -src_voc          FILE : src vocab (needed to initialize learning)
   -tgt_voc          FILE : tgt vocab (needed to initialize learning)
   -lid_voc          FILE : lid vocab (needed to initialize learning)

   -src_tok          FILE : src json tokenization options for onmt tokenization
   -tgt_tok          FILE : tgt json tokenization options for onmt tokenization

   Network:
   -net_wrd_len       INT : word src/tgt embeddings size [320]
   -net_conv_lens  STRING : kernel sizes of src convolutional layers [0] (not used)
   -net_blstm_lens STRING : units of src bi-lstm layers (separated by -) [1024-1024-1024] (3 layers with 512 cells for each direction)
   -net_sentence   STRING : how src sentence embedding is formed from previous layer: last, mean, max [max]
   -net_lid_len       INT : tgt lid embedding size [32]
   -net_lstm_len      INT : units of the tgt lstm layer [2048]
   -net_opt        STRING : GD method either: adam, adagrad, adadelta, sgd, rmsprop [adam]

   -dropout         FLOAT : dropout ratio applided to different layers [0.3]
   -opt_lr          FLOAT : initial learning rate [0.0002]
   -opt_decay       FLOAT : learning rate decay value when opt_method='sgd' [0.9] (not used for adam)
   -clip            FLOAT : gradient clipping value (0.0 for no clipping) [0.0]
   -max_sents         INT : Consider this number of sentences per batch (0 for all) [0]
   -n_epochs          INT : train for this number of epochs [1]
   -reports           INT : report every this many batches [100]

 [INFERENCE OPTIONS]
*  -epoch             INT : epoch to use ([mdir]/epoch[epoch] must exist)
*  -src_tst          FILE : src testing data
*  -tgt_tst          FILE : tgt testing data
   -show_sim              : output similarity score
   -show_oov              : output number of OOVs 
   -show_emb              : output sentence embeddings
   -show_snt              : output original sentences

+ Options marked with * must be set. The rest have default values.
+ If -mdir exists in learning mode, learning continues after restoring the last model
+ Training data is always shuffled at every epoch
+ -show_sim, -show_oov, -show_emb, -show_snt can be used at the same time
""".format(sys.argv.pop(0))

        self.mdir = None
        self.seq_size = 50
        self.batch_size = 32
        self.seed = 12345
        #files
        self.src_trn = None
        self.tgt_trn = None
        self.lid_trn = None
        self.src_val = None
        self.tgt_val = None
        self.lid_val = None
        self.src_voc = None
        self.tgt_voc = None
        self.lid_voc = None
        self.src_tok = None
        self.tgt_tok = None
        #will be created
        self.voc_src = None
        self.voc_tgt = None
        self.voc_lid = None
        self.emb_src = None
        self.emb_tgt = None
        self.emb_lid = None
        self.tok_src = None
        self.tok_tgt = None
        #network
        self.net_wrd_len = 320
        self.net_conv_lens = [0]
        self.net_blstm_lens = [1024, 1024, 1024]
        self.net_sentence = 'max'
        self.net_lid_len = 32
        self.net_lstm_len = 2048
        self.net_opt = 'adam'
        #optimization
        self.dropout = 0.3
        self.opt_lr = 0.0002
        self.opt_decay = 0.9
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

        self.parse(sys.argv)
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        if not self.mdir:
            sys.stderr.write("error: Missing -mdir option\n{}".format(self.usage))
            sys.exit()
        if self.src_tst: self.inference()
        elif self.src_trn and self.tgt_trn and self.src_val and self.tgt_val: self.learn()
        return

    def parse(self, argv):
        while len(argv):
            tok = argv.pop(0)
            if (tok=="-mdir" and len(argv)):             self.mdir = argv.pop(0)
            elif (tok=="-seq_size" and len(argv)):       self.seq_size = int(argv.pop(0))
            elif (tok=="-batch_size" and len(argv)):     self.batch_size = int(argv.pop(0))
            elif (tok=="-seed" and len(argv)):           self.seed = int(argv.pop(0))
            #files
            elif (tok=="-src_trn" and len(argv)):        self.src_trn = argv.pop(0)
            elif (tok=="-tgt_trn" and len(argv)):        self.tgt_trn = argv.pop(0)
            elif (tok=="-lid_trn" and len(argv)):        self.lid_trn = argv.pop(0)
            elif (tok=="-src_val" and len(argv)):        self.src_val = argv.pop(0)
            elif (tok=="-tgt_val" and len(argv)):        self.tgt_val = argv.pop(0)
            elif (tok=="-lid_val" and len(argv)):        self.lid_val = argv.pop(0)
            elif (tok=="-src_voc" and len(argv)):        self.src_voc = argv.pop(0)
            elif (tok=="-tgt_voc" and len(argv)):        self.tgt_voc = argv.pop(0)
            elif (tok=="-lid_voc" and len(argv)):        self.lid_voc = argv.pop(0)
            elif (tok=="-src_tok" and len(argv)):        self.src_tok = argv.pop(0)
            elif (tok=="-tgt_tok" and len(argv)):        self.tgt_tok = argv.pop(0)
            #network
            elif (tok=="-net_wrd_len" and len(argv)):    self.net_wrd_len = int(argv.pop(0))
            elif (tok=="-net_blstm_lens" and len(argv)): self.net_blstm_lens = map(int, argv.pop(0).split('-'))
            elif (tok=="-net_conv_lens" and len(argv)):  self.net_conv_lens = map(int, argv.pop(0).split('-'))
            elif (tok=="-net_sentence" and len(argv)):   self.net_sentence = argv.pop(0)
            elif (tok=="-net_lid_len" and len(argv)):    self.net_lid_len = int(argv.pop(0))
            elif (tok=="-net_lstm_len" and len(argv)):   self.net_lstm_len = int(argv.pop(0))
            elif (tok=="-net_opt" and len(argv)):        self.net_opt = argv.pop(0)
            #optimization
            elif (tok=="-dropout" and len(argv)):        self.dropout = float(argv.pop(0))
            elif (tok=="-opt_lr" and len(argv)):         self.opt_lr = float(argv.pop(0))
            elif (tok=="-opt_decay" and len(argv)):      self.opt_decay = float(argv.pop(0))
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
            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

    def inference(self):
        if not self.epoch:
            sys.stderr.write("error: Missing -epoch option\n{}".format(self.usage))
            sys.exit()
        if not os.path.exists(self.src_tst):
            sys.stderr.write('error: -src_tst file {} cannot be find\n{}'.format(self.src_tst,self.usage))
            sys.exit()
#        if not os.path.exists(self.tgt_tst):
#            sys.stderr.write('error: -tgt_tst file {} cannot be find\n{}'.format(self.tgt_tst,self.usage))
#            sys.exit()
        if not os.path.exists(self.mdir + '/epoch' + self.epoch + '.index'):
            sys.stderr.write('error: -epoch file {} cannot be find\n{}'.format(self.mdir + '/epoch' + self.epoch + '.index',self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/topology'): 
            sys.stderr.write('error: topology file: {} cannot be find\n{}'.format(self.mdir + '/topology',self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/vocab_src'): 
            sys.stderr.write('error: vocab_src file: {} cannot be find\n{}'.format(self.mdir + '/vocab_src',self.usage))
            sys.exit()
        if not os.path.exists(self.mdir + '/vocab_tgt'): 
            sys.stderr.write('error: vocab_tgt file: {} cannot be find\n{}'.format(self.mdir + '/vocab_tgt',self.usage))
            sys.exit()
        argv = []
        with open(self.mdir + "/topology", 'r') as f:
            for line in f:
                opt, val = line.split()
                argv.append('-'+opt)
                argv.append(val)
        self.parse(argv) ### this overrides options passed in command line
        self.dropout = 0.0
        self.seq_size = 0

        ### read vocabularies
        self.voc_src = Vocab(self.mdir + "/vocab_src", True)
        self.voc_tgt = Vocab(self.mdir + "/vocab_tgt", True)
        if os.path.exists(self.mdir + '/token_src'): 
            self.src_tok = self.mdir + '/token_src'
            with open(self.mdir + '/token_src') as jsonfile: 
                tok_src_opt = json.load(jsonfile)
                tok_src_opt["vocabulary"] = self.mdir + '/vocab_src'
                self.tok_src = build_tokenizer(tok_src_opt)
        if os.path.exists(self.mdir + '/token_tgt'): 
            self.tgt_tok = self.mdir + '/token_tgt'
            with open(self.mdir + '/token_tgt') as jsonfile: 
                tok_tgt_opt = json.load(jsonfile)
                tok_tgt_opt["vocabulary"] =  self.mdir + '/vocab_tgt'
                self.tok_tgt = build_tokenizer(tok_tgt_opt)


        return  

    def learn(self):
        if not os.path.exists(self.src_trn):
            sys.stderr.write('error: -src_trn file {} cannot be find\n{}'.format(self.src_trn,self.usage))
            sys.exit()
        if not os.path.exists(self.tgt_trn):
            sys.stderr.write('error: -tgt_trn file {} cannot be find\n{}'.format(self.tgt_trn,self.usage))
            sys.exit()
        if not os.path.exists(self.lid_trn):
            sys.stderr.write('error: -lid_trn file {} cannot be find\n{}'.format(self.lid_trn,self.usage))
            sys.exit()
        if not os.path.exists(self.src_val):
            sys.stderr.write('error: -src_val file {} cannot be find\n{}'.format(self.src_val,self.usage))
            sys.exit()
        if not os.path.exists(self.tgt_val):
            sys.stderr.write('error: -tgt_val file {} cannot be find\n{}'.format(self.tgt_val,self.usage))
            sys.exit()
        if not os.path.exists(self.lid_val):
            sys.stderr.write('error: -lid_val file {} cannot be find\n{}'.format(self.lid_val,self.usage))
            sys.exit()
        ###
        ### continuation
        ###
        if os.path.exists(self.mdir): 
            if not os.path.exists(self.mdir + '/topology'): 
                sys.stderr.write('error: topology file: {} cannot be find\n{}'.format(self.mdir + '/topology',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab_src'): 
                sys.stderr.write('error: vocab_src file: {} cannot be find\n{}'.format(self.mdir + '/vocab_src',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab_tgt'): 
                sys.stderr.write('error: vocab_tgt file: {} cannot be find\n{}'.format(self.mdir + '/vocab_tgt',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab_lid'): 
                sys.stderr.write('error: vocab_lid file: {} cannot be find\n{}'.format(self.mdir + '/vocab_lid',self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/checkpoint'): 
                sys.stderr.write('error: checkpoint file: {} cannot be find\ndelete dir {} ???\n{}'.format(self.mdir + '/checkpoint', self.mdir,self.usage))
                sys.exit()

            argv = []
            with open(self.mdir + "/topology", 'r') as f:
                for line in f:
                    opt, val = line.split()
                    argv.append('-'+opt)
                    argv.append(val)
            self.parse(argv) ### this overrides options passed in command line

            ### read vocabularies
            self.voc_src = Vocab(self.mdir + "/vocab_src", True) 
            self.voc_tgt = Vocab(self.mdir + "/vocab_tgt", True)
            self.voc_lid = Vocab(self.mdir + "/vocab_lid", False)

            ### use existing tokenizers if exist
            if os.path.exists(self.mdir + '/token_src'): 
                self.src_tok = self.mdir + '/token_src'
                with open(self.mdir + '/token_src') as jsonfile: 
                    tok_src_opt = json.load(jsonfile)
                    tok_src_opt["vocabulary"] = self.mdir + '/vocab_src'
                    self.tok_src = build_tokenizer(tok_src_opt)

            if os.path.exists(self.mdir + '/token_tgt'): 
                self.tgt_tok = self.mdir + '/token_tgt'
                with open(self.mdir + '/token_tgt') as jsonfile: 
                    tok_tgt_opt = json.load(jsonfile)
                    tok_tgt_opt["vocabulary"] = self.mdir + '/vocab_tgt'
                    self.tok_tgt = build_tokenizer(tok_tgt_opt)

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
            if not os.path.exists(self.mdir): 
                os.makedirs(self.mdir)

            #copy vocabularies
            copyfile(self.src_voc, self.mdir + "/vocab_src")
            copyfile(self.tgt_voc, self.mdir + "/vocab_tgt")
            copyfile(self.lid_voc, self.mdir + "/vocab_lid")
            #copy tokenizers if exist
            if self.src_tok: copyfile(self.src_tok, self.mdir + "/token_src")
            if self.tgt_tok: copyfile(self.tgt_tok, self.mdir + "/token_tgt")

            ### read vocabularies
            self.voc_src = Vocab(self.mdir + "/vocab_src", True) 
            self.voc_tgt = Vocab(self.mdir + "/vocab_tgt", True)
            self.voc_lid = Vocab(self.mdir + "/vocab_lid", False)

            ### use existing tokenizers if exist
            if os.path.exists(self.mdir + '/token_src'): 
                self.src_tok = self.mdir + '/token_src'
                with open(self.mdir + '/token_src') as jsonfile: 
                    tok_src_opt = json.load(jsonfile)
                    tok_src_opt["vocabulary"] = self.mdir + '/vocab_src'
                    self.tok_src = build_tokenizer(tok_src_opt)

            if os.path.exists(self.mdir + '/token_tgt'): 
                self.tgt_tok = self.mdir + '/token_tgt'
                with open(self.mdir + '/token_tgt') as jsonfile: 
                    tok_tgt_opt = json.load(jsonfile)
                    tok_tgt_opt["vocabulary"] = self.mdir + '/vocab_tgt'
                    self.tok_tgt = build_tokenizer(tok_tgt_opt)

            #create embeddings
            self.emb_src = Embeddings(self.voc_src,self.net_wrd_len)
            self.emb_tgt = Embeddings(self.voc_tgt,self.net_wrd_len)
            self.emb_lid = Embeddings(self.voc_lid,self.net_lid_len)

            #write topology file
            with open(self.mdir + "/topology", 'w') as f: 
                for opt, val in vars(self).items():
                    if not opt.startswith("net"): continue
                    if opt.endswith("_lens"):
                        if len(val)>0: 
                            sval = "-".join([str(v) for v in val])
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
                if name=="usage" or name.startswith("emb_") or name.startswith("voc_") or name.startswith("tok_"): continue
                f.write("{} {}\n".format(name,val))



