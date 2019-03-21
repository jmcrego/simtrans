import tensorflow as tf
import numpy as np
import io
import os
import sys
import yaml
from shutil import copyfile
from vocab import Vocab
from tokenizer import build_tokenizer

class network():

    def __init__(self, file):
        if not os.path.exists(file): 
            sys.stderr.write('error: missing network file={}\n'.format(file))
            sys.exit()

        self.enc = None
        self.ali = None
        self.trn = None
        self.dir = None
        with open(file, 'r') as f:
            for line in f:
                opt, val = line.strip('\n').split()
                if opt == 'enc': self.enc = val
                elif opt == 'ali': self.ali = val
                elif opt == 'trn': self.trn = val
                elif opt == 'dir': self.dir = val
                elif opt == 'opt': self.opt = val
                elif opt == 'lid': self.lid = val
                else:
                    sys.stderr.write('error: bad network option={}\n'.format(opt))
                    sys.exit()

        self.enc_layers = []
        self.ali_layers = []
        self.trn_layers = []
        self.dir_layers = []
        if self.enc is not None:
            self.enc_layers = self.enc.split(',')
        if self.ali is not None:
            self.ali_layers = self.ali.split(',')
        if self.trn is not None:
            self.trn_layers = self.trn.split(',')
        if self.dir is not None:
            self.dir_layers = self.dir.split(',')
        sys.stderr.write('Read Network {}\n'.format(file))

    def nlayers(self, which):
        if which=='enc': return len(self.enc_layers)
        elif which=='ali': return len(self.ali_layers)
        elif which=='trn': return len(self.trn_layers)
        elif which=='dir': return len(self.dir_layers)
        else:
            sys.stderr.write('error: unknown layer={}\n'.format(which))
            sys.exit()

    def layer(self, which, l):
        if which=='enc':
            if l>=len(self.enc_layers):
                sys.stderr.write('error: bad enc layer index={}\n'.format(l))
                sys.exit()
            fields = self.enc_layers[l].split('-')
        elif which=='ali':
            if l>=len(self.ali_layers):
                sys.stderr.write('error: bad ali layer index={}\n'.format(l))
                sys.exit()
            fields = self.ali_layers[l].split('-')
        elif which=='trn':
            if l>=len(self.trn_layers):
                sys.stderr.write('error: bad trn layer index={}\n'.format(l))
                sys.exit()
            fields = self.trn_layers[l].split('-')
        elif which=='dir':
            if l>=len(self.dir_layers):
                sys.stderr.write('error: bad dir layer index={}\n'.format(l))
                sys.exit()
            return int(self.dir_layers[l])
        else:
            sys.stderr.write('error: unknown layer={}\n'.format(which))
            sys.exit()

        ### returns: type, filters, units, dropout, name
        if   fields[0].lower() == 'w': return fields[0], 0,              int(fields[1]), float(fields[2]), fields[3] #w-256-0.3-name
        elif fields[0].lower() == 'c': return fields[0], int(fields[1]), int(fields[2]), float(fields[3]), fields[4] #c-3-256-0.3-name
        elif fields[0].lower() == 'b': return fields[0], 0,              int(fields[1]), float(fields[2]), fields[3] #b-256-0.3-name
        elif fields[0].lower() == 'l': return fields[0], 0,              int(fields[1]), float(fields[2]), fields[3] #l-256-0.3-name
        elif fields[0].lower() == 's': return fields[0], 0,              0,              0.0,              fields[1] #s-last
        else:
            sys.stderr.write('error: bad layer type={}\n'.format(fields[0]))
            sys.exit()

    def write(self, file):
        with open(file, 'w') as f: 
            f.write("enc {}\n".format(self.enc))
            if self.ali:
                f.write("ali {}\n".format(self.ali))
            if self.trn:
                f.write("trn {}\n".format(self.trn))
            if self.dir:
                f.write("dir {}\n".format(self.dir))
            f.write("opt {}\n".format(self.opt))
            f.write("lid {}\n".format(self.lid))



class Config():

    def __init__(self, argv):
        self.usage="""usage: {}
*  -mdir              DIR : directory to save/restore models
   -batch_size        INT : number of examples per batch [32]
   -seed              INT : seed for randomness [12345]
   -h                     : extended help message

 [LEARNING OPTIONS]
*  -src_trn          FILE : src training files (comma-separated)
*  -tgt_trn          FILE : tgt training files (comma-separated)
*  -tgt_trn_lid      FILE : lid of tgt training files (comma-separated)

*  -src_val          FILE : src validation files (comma-separated)
*  -tgt_val          FILE : tgt validation files (comma-separated)
*  -tgt_val_lid      FILE : lid of tgt validation files (comma-separated)

   -voc              FILE : src/tgt vocab file (needed to initialize learning)
   -tok              FILE : src/tgt onmt tok options yaml file (needed to initialize learning)
   -net              FILE : network topology file (needed to initialize learning)

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
""".format(sys.argv.pop(0))

        self.help="""+ Options marked with * must be set. The rest have default values.
+ If -mdir exists in learning mode, learning continues after restoring the last model
+ Training data is always shuffled at every epoch

====== Tokenization options (yaml) example ======
mode: aggressive
joiner_annotate: True
segment_numbers: True
segment_case: True
segment_alphabet: True
bpe_model_path: file

====== Network topology options example ======
enc w-256-0.3-both,c-3-512-0.3-src,b-512-0.3-src,b-512-0.3-src,s-last
ali w-256-0.3-both,c-3-512-0.3-tgt,b-512-0.3-tgt,b-512-0.3-tgt,s-last
trn w-256-0.3-both,l-1024-0.3-tgt
dir 1024
opt adam
lid English,French,German

- enc layer is always used
- at least one ali/trn layer must be used (both are also alowed)

enc/ali layers (comma-separated list):
    + The first is word embedding layer
    + The last indicates how the final state is built (last, mean, max)     
    + Layers follow the next formats:
        + Wembedding    (w-embedSize-dropout-name)
        + Convolutional (c-filters-kernelSize-dropout-name)
        + Bi-lstm       (b-hiddenSize-dropout-name) 
        + Lstm          (l-hiddenSize-dropout-name)
        + Sembedding    (s-finalState)

trn layers (comma-separated list):
    use WEmbedding and Lstm/Bi-lstm layers  (Ex: w-256-0.3-name,l-2048-0.2-name)

dir layers (comma-separated list):
    use 0 for not using a dense layer

Network optimization:
    + adam
    + adagrad
    + adadelta
    + sgd
    + rmsprop

Network LIDs (comma-separated list)
"""

        self.mdir = None
        self.max_seq_size = 50
        self.batch_size = 32
        self.seed = 12345
        #files
        self.src_trn = []
        self.tgt_trn = []
        self.tgt_trn_lid = []
        self.src_val = []
        self.tgt_val = []
        self.tgt_val_lid = []
        self.src_tst = None
        self.tgt_tst = None
        self.voc = None
        self.tok = None
        self.net = None
        #will be created
        self.vocab = None #vocabulary
        self.token = None #onmt tokenizer
        self.network = None #network topology
        #optimization
        self.opt_lr = 0.0001
        self.opt_decay = 0.96
        self.opt_minlr = 0.0
        self.clip = 0.0
        self.max_sents = 0
        self.n_epochs = 1 # epochs to run
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
        self.is_inference = False
        if self.src_tst is not None:
            self.is_inference = True
            self.max_seq_size = 0
            self.max_sents = 0

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.create_dir_copy_files()

        ### read network
        self.network = network(self.net)
        ### read vocabulary
        self.vocab = Vocab(self.voc, self.network.lid)
        ### read tokenizer
        if os.path.exists(self.tok): 
            with open(self.tok) as yamlfile: 
                tok_opt = yaml.load(yamlfile)
                self.token = build_tokenizer(tok_opt)

        #################
        ### inference ###
        #################
        if self.is_inference:
            if not self.epoch:
                sys.stderr.write("error: Missing -epoch option\n{}".format(self.usage))
                sys.exit()
            if not os.path.exists(self.mdir + '/epoch' + self.epoch + '.index'):
                sys.stderr.write('error: -epoch file {} cannot be find\n{}'.format(self.mdir + '/epoch' + self.epoch + '.index',self.usage))
                sys.exit()
            self.max_seq_size = 0
            self.max_sents = 0
            return

        ################
        ### learning ###
        ################
        if not len(self.src_trn) or len(self.src_trn)!=len(self.tgt_trn) or len(self.src_trn)!=len(self.tgt_trn_lid):
            sys.stderr.write('error: bad training files\n{}'.format(self.usage))
            sys.exit()

        elif not len(self.src_val) or len(self.src_val)!=len(self.tgt_val) or len(self.src_val)!=len(self.tgt_val_lid): 
            sys.stderr.write('error: bad validation files\n{}'.format(self.usage))
            sys.exit()

        if os.path.exists(self.mdir + '/checkpoint'): 
            ### continuation (update last epoch)
            for e in range(999,0,-1):
                if os.path.exists(self.mdir+"/epoch{}.index".format(e)): 
                    self.last_epoch = e
                    break
            sys.stderr.write("learning continuation: last epoch is {}\n".format(self.last_epoch))

        else:
            sys.stderr.write("learning from scratch\n")


    def create_dir_copy_files(self):

        if os.path.exists(self.mdir + '/checkpoint') and (self.voc is not None or self.tok is not None or self.net is not None): 
            sys.stderr.write('error: replacing voc/tok/net while exists previous {}/checkpoint\n'.format(self.mdir))
            sys.exit()

        if not os.path.exists(self.mdir):
            os.makedirs(self.mdir)
            sys.stderr.write('created mdir={}\n'.format(self.mdir))
        else:
            sys.stderr.write('reusing mdir={}\n'.format(self.mdir))


        if self.voc is not None:
            copyfile(self.voc, self.mdir + "/vocab")
            sys.stderr.write('copied voc file={}\n'.format(self.voc))
    
        if self.net is not None:
            copyfile(self.net, self.mdir + "/network")
            sys.stderr.write('copied net file={}\n'.format(self.net))

        if self.tok is not None: 
            with open(self.tok) as f: 
                tok_opt = yaml.load(f)
                ### replaces/creates vocab option in token
                tok_opt["vocabulary"] = self.mdir + '/vocab'
                ### if exists bpe_model_path option, copy model to mdir and replaces bpe_model_path in token
                if 'bpe_model_path' in tok_opt:
                    copyfile(tok_opt['bpe_model_path'], self.mdir + "/bpe")
                    sys.stderr.write('copied bpe file={}\n'.format(tok_opt['bpe_model_path']))
                    tok_opt['bpe_model_path'] = self.mdir + '/bpe'
                ### dump token to mdir
                with open(self.mdir + '/token', 'w') as outfile: 
                    yaml.dump(tok_opt, outfile, default_flow_style=True)      
                sys.stderr.write('copied tok file={}\n'.format(self.tok))

        self.voc = self.mdir + "/vocab"
        self.net = self.mdir + "/network"
        self.tok = self.mdir + "/token"

        if not os.path.exists(self.voc): 
            sys.stderr.write('error: cannot find vocabulary file\n{}'.format(self.usage))
            sys.exit()
        if not os.path.exists(self.net): 
            sys.stderr.write('error: cannot find network file\n{}'.format(self.usage))
            sys.exit()


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
            elif (tok=="-net" and len(argv)):            self.net = argv.pop(0)
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
                sys.stderr.write("{}\n{}\n".format(self.usage,self.help))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

        if not self.mdir:
            sys.stderr.write("error: Missing -mdir option\n{}".format(self.usage))
            sys.exit()
        if len(self.mdir)>1 and self.mdir[-1]=="/": self.mdir = self.mdir[0:-1] ### delete ending '/'


    def write_config(self):
        file = self.mdir + "/epoch"+str(self.last_epoch)+".config"
        with open(file,"w") as f:
            for name, val in vars(self).items():
                if name=="usage" or name.startswith("voc") or name.startswith("tok") or name.startswith("net"): continue
                f.write("{} {}\n".format(name,val))


