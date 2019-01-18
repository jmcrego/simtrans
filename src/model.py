# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
import time
from random import randint
from config import Config
#from dataset import minibatches

def GetHumanReadable(size,precision=2):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])

class Score():
    def __init__(self):
        #loss
        self.sumloss = 0.0
        self.total_iters = 0
        self.Loss = 0.0
        #tokens
        self.sumok = 0
        self.total_words = 0
        self.Acc = 0.0

    def add(self,loss,tout,tref,tmask):
        self.total_iters += 1
        self.sumloss += loss
        self.Loss = self.sumloss / self.total_iters
        if len(tout):
            masked_equals = np.logical_and(np.equal(tout,tref), tmask)
            self.total_words += np.sum(tmask)
            self.sumok += np.count_nonzero(masked_equals)
            self.Acc = 100.0 * self.sumok / self.total_words


class Model():
    def __init__(self, config):
        self.config = config
        self.sess = None

    def embedding_initialize(self,NS,ES,embeddings):
        if embeddings is not None: 
            m = embeddings.matrix
        else:
            sys.stderr.write("embeddings randomly initialized\n")
            m = tf.random_uniform([NS, ES], minval=-0.1, maxval=0.1)
        return m

###################
### build graph ###
###################

    def add_placeholders(self):
        self.input_src     = tf.placeholder(tf.int32, shape=[None,None], name="input_src")  # Shape: batch_size x |Fj| (sequence length)
        self.input_tgt     = tf.placeholder(tf.int32, shape=[None,None], name="input_tgt")  # Shape: batch_size x |Ei| (sequence length)
        self.input_ref     = tf.placeholder(tf.int32, shape=[None,None], name="input_ref")  # Shape: batch_size x |Ei| (sequence length)
        self.len_src       = tf.placeholder(tf.int32, shape=[None],      name="len_src")
        self.len_tgt       = tf.placeholder(tf.int32, shape=[None],      name="len_tgt")
        self.lr            = tf.placeholder(tf.float32, shape=[],        name="lr")

    def add_encoder(self):
        K = 1.0-self.config.dropout   # keep probability for embeddings dropout Ex: 0.7
        B = tf.shape(self.input_src)[0] #batch size
        Ss = tf.shape(self.input_src)[1] #seq_length
        Vs = self.config.voc_src.length #src vocab
        Es = self.config.net_wrd_len #src embedding size
        Hs = np.divide(self.config.net_blstm_lens, 2) #src lstm sizes (half cells for each direction)

        with tf.device('/cpu:0'), tf.variable_scope("embedding_src",reuse=tf.AUTO_REUSE):
            self.LT_src = tf.get_variable(initializer = self.embedding_initialize(Vs, Es, self.config.emb_src), dtype=tf.float32, name="embeddings_src")
            self.embed_src = tf.nn.embedding_lookup(self.LT_src, self.input_src, name="embed_src")
            self.embed_src = tf.nn.dropout(self.embed_src, keep_prob=K)  #[B,Ss,Es]

        self.out_src = self.embed_src #[B,Ss,Es]
        #if not bi-lstm layers are used sentence_src is computed using either: max, mean (not last)
        if len(Hs)>0 and Hs[0]>0:
            for l in range(len(Hs)):
                with tf.variable_scope("blstm_src_{}".format(l),reuse=tf.AUTO_REUSE):
                    cell_fw = tf.contrib.rnn.LSTMCell(Hs[l], initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(Hs[l], initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, input_keep_prob=K)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, input_keep_prob=K)      
                    (output_src_fw, output_src_bw), (last_src_fw, last_src_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.out_src, sequence_length=self.len_src, dtype=tf.float32)
                    self.out_src = tf.concat([output_src_fw, output_src_bw], axis=2)  #[B,Ss,Hs*2]
            self.last_src = tf.concat([last_src_fw[1], last_src_bw[1]], axis=1) #[B, Hs*2] (i take [1] since last_state is a tuple with (c,h))

        with tf.variable_scope("sentence_src"):
            if self.config.net_sentence == 'last':
                if len(Hs) == 0: 
                    sys.stderr.write("error: -net_sentence 'last' cannot be used without any -net_blstm_lens layers\n")
                    sys.exit()
                self.embed_snt = self.last_src #[B,Hs*2]
            elif self.config.net_sentence == 'max':
                mask = tf.expand_dims(tf.sequence_mask(self.len_src, dtype=tf.float32), 2)
                self.embed_snt = self.out_src * mask + (1-mask) * tf.float32.min #masked tokens contain -Inf
                self.embed_snt = tf.reduce_max(self.embed_snt, axis=1) #[B,Hs*2] or [B,Es] if not bi-lstm layers
            elif self.config.net_sentence == 'mean':
                mask = tf.expand_dims(tf.sequence_mask(self.len_src, dtype=tf.float32), 2) #[B, Ss] => [B, Ss, 1]
                self.embed_snt = self.out_src * mask #masked tokens contain 0.0
                self.embed_snt = tf.reduce_sum(self.embed_snt, axis=1) / tf.expand_dims(tf.to_float(self.len_src), 1) #[B,Hs*2]
            else:
                sys.stderr.write("error: bad -net_sentence option '{}'\n".format(self.config.net_sentence))
                sys.exit()


    def add_decoder(self):
        K = 1.0-self.config.dropout # keep probability for embeddings dropout Ex: 0.7
        B = tf.shape(self.input_tgt)[0] #batch size
        St = tf.shape(self.input_tgt)[1] #seq_length
        Vt = self.config.voc_tgt.length #tgt vocab
        Et = self.config.net_wrd_len #tgt embedding size
        Ht = self.config.net_lstm_len #tgt lstm size

        with tf.variable_scope("embed_snt2lstm_initial",reuse=tf.AUTO_REUSE):
            initial_state_h = tf.layers.dense(self.embed_snt, Ht, use_bias=False) # Hs*2 => Ht
            initial_state_c = tf.zeros(tf.shape(initial_state_h))
            self.initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_c, initial_state_h)

        with tf.device('/cpu:0'), tf.variable_scope("embedding_tgt"):
            self.LT_tgt = tf.get_variable(initializer = self.embedding_initialize(Vt, Et, self.config.emb_tgt), dtype=tf.float32, name="embeddings_tgt")
            self.embed_tgt = tf.nn.embedding_lookup(self.LT_tgt, self.input_tgt, name="embed_tgt") #[B, St, Et]
            self.embed_tgt = tf.nn.dropout(self.embed_tgt, keep_prob=K)

        with tf.variable_scope("lstm_tgt",reuse=tf.AUTO_REUSE):
            self.embed_snt_src = tf.expand_dims(self.embed_snt, 1) #[B,Hs*2] or [B,Es] => [B,1,Hs*2] or [B,1,Es]
            self.embed_snt_src = tf.tile(self.embed_snt_src, [1, St, 1]) #[B,St,Hs*2] or [B,St,Es]
            self.embed_snt_src_plus_tgt = tf.concat([self.embed_snt_src, self.embed_tgt], 2) #[B, St, Hs*2+Et] or [B, St, Es+Et]
            cell = tf.contrib.rnn.LSTMCell(Ht)
            self.out_tgt, state_tgt = tf.nn.dynamic_rnn(cell, self.embed_snt_src_plus_tgt, initial_state=self.initial_state, sequence_length=self.len_tgt, dtype=tf.float32)
            ### self.embed_tgt is like: LID my sentence <pad> ... (LID is like a bos that also encodes the language to produce)
            ### self.out_tgt   is like: my sentence <eos> 0.0 ...

        with tf.variable_scope("logits",reuse=tf.AUTO_REUSE):
            self.out_logits = tf.layers.dense(self.out_tgt, Vt)
            self.out_pred = tf.argmax(self.out_logits, 2)

        pars = sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())
        sys.stderr.write("Total Enc/Dec parameters: {} => {}\n".format(pars, GetHumanReadable(pars*4))) #one parameter is 4 bytes (float32)
        for var in tf.trainable_variables(): 
            pars = var.get_shape().num_elements()
            sys.stderr.write("\t{} => {} {}\n".format(pars, GetHumanReadable(pars*4), var))


    def add_loss(self):
        Vt = self.config.voc_tgt.length #tgt vocab

        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_ref, depth=Vt, dtype=tf.float32), logits=self.out_logits) #[B, S]
            self.tmask = tf.sequence_mask(self.len_tgt, dtype=tf.float32) #[B, S]            
            self.loss = tf.reduce_sum(xentropy*self.tmask) / tf.to_float(tf.reduce_sum(self.len_tgt))


    def add_train(self):
        if   self.config.net_opt == 'adam':     self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.config.net_opt == 'sgd':      self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.config.net_opt == 'adagrad':  self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.config.net_opt == 'rmsprop':  self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.config.net_opt == 'adadelta': self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        else:
            sys.stderr.write("error: bad -net_opt option '{}'\n".format(self.config.net_opt))
            sys.exit()

        if self.config.clip > 0.0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.train_op = self.optimizer.minimize(self.loss)


    def build_graph(self):
        self.add_placeholders()
        self.add_encoder()  
        if self.config.src_tst is None: 
            self.add_decoder()
            self.add_loss()
            self.add_train()

###################
### feed_dict #####
###################

    def get_feed_dict(self, src, len_src, tgt=[[]], len_tgt=[], ref=[[]], lr=0.0):
        feed = { 
            self.input_src: src,
            self.input_tgt: tgt,
            self.input_ref: ref,
            self.len_src: len_src,
            self.len_tgt: len_tgt,
            self.lr: lr
        }
        return feed

###################
### learning ######
###################

    def run_epoch(self, train, dev, lr):
        #######################
        # learn on trainset ###
        #######################
        len_train = train.len
        if self.config.max_sents: len_train = min(len_train, self.config.max_sents)
        nbatches = (len_train + self.config.batch_size - 1) // self.config.batch_size
        curr_epoch = self.config.last_epoch + 1
        ini_time = time.time()
        score = Score()
        pscore = Score() ### partial score
        tpre = time.time()
        for iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch) in enumerate(train):
            assert(np.array(tgt_batch).shape == np.array(ref_batch).shape)
            fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch, lr)
            #embed_src, out_src, initial_state, embed_snt, embed_tgt, embed_snt_tgt, out_tgt, out_logits, out_pred = self.sess.run([self.embed_src, self.out_src, self.initial_state, self.embed_snt, self.embed_tgt, self.embed_snt_tgt, self.out_tgt, self.out_logits, self.out_pred], feed_dict=fd)
            #print("shape of embed_src = {}".format(np.array(embed_src).shape))
            #print("shape of out_src = {}".format(np.array(out_src).shape))
            #print("shape of initial_state = {}".format(np.array(initial_state).shape))
            #print("shape of embed_snt = {}".format(np.array(embed_snt).shape))
            #print("shape of embed_tgt = {}".format(np.array(embed_tgt).shape))
            #print("shape of embed_snt_tgt = {}".format(np.array(embed_snt_tgt).shape))
            #print("shape of out_tgt = {}".format(np.array(out_tgt).shape))
            #print("shape of out_logits = {}".format(np.array(out_logits).shape))
            #print("shape of out_pred = {}\n{}".format(np.array(out_pred).shape, out_pred))
            #sys.exit()
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            score.add(loss,[],[],[])
            pscore.add(loss,[],[],[])
            if (iter+1)%self.config.reports == 0:
                tnow = time.time()
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                sys.stderr.write('{} Epoch {} Iteration {}/{} (loss={:.6f}) lr={:.6f} time={:.2f} sec\n'.format(curr_time,curr_epoch,iter+1,nbatches,pscore.Loss,lr,tnow-tpre))
                tpre = tnow
                pscore = Score()
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write('{} Epoch {} TRAIN (loss={:.4f})'.format(curr_time,curr_epoch,score.Loss))
        sys.stderr.write(' Train set: words={}/{} %oov={:.2f}/{:.2f}\n'.format(train.nsrc, train.ntgt, 100.0*train.nunk_src/train.nsrc, 100.0*train.nunk_tgt/train.ntgt))
        #keep records
        self.config.tloss = score.Loss
        self.config.time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        self.config.seconds = "{:.2f}".format(time.time() - ini_time)
        self.config.last_epoch += 1
        self.save_session(self.config.last_epoch)

        ##########################
        # evaluate over devset ###
        ##########################
        score = Score()
        if dev is not None:
            nbatches = (dev.len + self.config.batch_size - 1) // self.config.batch_size
            for iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch) in enumerate(dev):
                fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch)
                loss, out_pred, tmask = self.sess.run([self.loss,self.out_pred,self.tmask], feed_dict=fd)
                score.add(loss,out_pred,ref_batch,tmask)
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            sys.stderr.write('{} Epoch {} VALID (loss={:.4f} Acc={:.2f})'.format(curr_time,curr_epoch,score.Loss,score.Acc))
            sys.stderr.write(' Valid set: words={}/{} %oov={:.2f}/{:.2f}\n'.format(dev.nsrc, dev.ntgt, 100.0*dev.nunk_src/dev.nsrc, 100.0*dev.nunk_tgt/dev.ntgt))
            #keep records
            self.config.vloss = score.Loss

        self.config.write_config()
        return score.Loss, curr_epoch


    def learn(self, train, dev, n_epochs):
        lr = self.config.opt_lr ### initial lr
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write("{} Start Training\n".format(curr_time))
        best_score = 0
        best_epoch = 0
        for iter in range(n_epochs):
            score, epoch = self.run_epoch(train, dev, lr)  ### decay when score does not improve over the best
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            if iter == 0 or score <= best_score: ### keep lr value
                best_score = score
                best_epoch = epoch
            elif lr >= self.config.opt_minlr: ### decay lr if score does not improve over the best score (and lr is not too low)
                lr *= self.config.opt_decay 
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write("{} End Training\n".format(curr_time))

###################
### inference #####
###################

    def inference(self, tst):
        nbatches = (tst.len + self.config.batch_size - 1) // self.config.batch_size

        ini_time = time.time()
        for iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch) in enumerate(tst):
            if len(tgt_batch[0]): bitext = True
            else: bitext = False
            #print("src_batch {}".format(src_batch))
            #print("len_src_batch {}".format(len_src_batch))
            #print("tgt_batch {}".format(tgt_batch))
            #print("len_tgt_batch {}".format(len_tgt_batch))
            #print("ref_batch {}".format(ref_batch))

            fd = self.get_feed_dict(src_batch, len_src_batch)
            embed_snt_src_batch = self.sess.run(self.embed_snt, feed_dict=fd)

            if bitext:
                tgt_batch_as_src, len_tgt_batch_as_src = self.ref_as_src(ref_batch, len_tgt_batch, tst.tgt_contains_lid)
                fd = self.get_feed_dict(tgt_batch_as_src, len_tgt_batch_as_src)
                embed_snt_tgt_batch = self.sess.run(self.embed_snt, feed_dict=fd)

            for i_sent in range(len(embed_snt_src_batch)):
                result = []
                if self.config.show_sim:
                    if bitext: result.append("{:.6f}".format(self.compute_sim(embed_snt_src_batch[i_sent], embed_snt_tgt_batch[i_sent])))

                if self.config.show_oov:
                    result.append("{}".format(nsrc_unk_batch[i_sent]))
                    if bitext: result.append("{}".format(ntgt_unk_batch[i_sent]))

                if self.config.show_emb: 
                    result.append(" ".join(["{:.6f}".format(e) for e in embed_snt_src_batch[i_sent]]))
                    if bitext: result.append(" ".join(["{:.6f}".format(e) for e in embed_snt_tgt_batch[i_sent]]))

                if self.config.show_snt: 
                    result.append(" ".join(raw_src_batch[i_sent]))
                    if bitext: result.append(" ".join(raw_tgt_batch[i_sent]))

                if self.config.show_idx: 
                    result.append(" ".join([str(e) for e in src_batch[i_sent]]))
                    if bitext: result.append(" ".join([str(e) for e in tgt_batch_as_src[i_sent]]))

                print "\t".join(result)

        end_time = time.time()
        sys.stderr.write("Analysed {} sentences with {} src tokens in {:.2f} seconds (model/test loading times not considered)\n".format(tst.len, tst.nsrc, end_time - ini_time))

    def ref_as_src(self, ref, len_ref, contains_lid):
        ### add in ref <bos>, incrase len_ref by 1
        ### it only works if both sides (src/tgt) have been seen by the encoder (sharing vocabularies)
        if contains_lid: #delete LID tokens
            ref = np.delete(ref, 0, 1) ### deletes the 0-th element in axis=1 from matrix ref
        else: #increase length by 1
            len_ref += np.ones_like(len_ref, dtype=int)

        #insert idx_bos in the begining (0) of axis=1 from matrix ref
#        ref = np.insert(ref, 0, self.config.voc_tgt.idx_bos, axis=1)
        for b in range(len(ref)):
            ref[b].insert(0,self.config.voc_tgt.idx_bos)

        return ref, len_ref

    def compute_sim(self, src, tgt):
        sim = np.sum((src/np.linalg.norm(src)) * (tgt/np.linalg.norm(tgt))) 
        return sim

###################
### session #######
###################

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=20)

        if self.config.epoch is not None: ### restore a file for testing
            fmodel = self.config.mdir + '/epoch' + self.config.epoch
            sys.stderr.write("Restoring model: {}\n".format(fmodel))
            self.saver.restore(self.sess, fmodel)
            return

        if self.config.mdir: ### initialize for training or restore previous
            if not os.path.exists(self.config.mdir + '/checkpoint'): 
                sys.stderr.write("Initializing model\n")
                self.sess.run(tf.global_variables_initializer())
            else:
                sys.stderr.write("Restoring previous model: {}\n".format(self.config.mdir))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.mdir))

    def save_session(self,e):
        if not os.path.exists(self.config.mdir): os.makedirs(self.config.mdir)
        file = "{}/epoch{}".format(self.config.mdir,e)
        self.saver.save(self.sess, file) #, max_to_keep=4, write_meta_graph=False) # global_step=step, keep_checkpoint_every_n_hours=2

    def close_session(self):
        self.sess.close()


