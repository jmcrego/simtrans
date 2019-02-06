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

    def wembedding(self, input, V, layer):
        cfg = layer.split('-')
        E = int(cfg[0])
        K = 1.0 - float(cfg[1])
        with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=tf.AUTO_REUSE): ### same embeddings for src/tgt words
            self.LT = tf.get_variable(initializer = tf.random_uniform([V, E], minval=-0.1, maxval=0.1), dtype=tf.float32, name="LT")
            embedded = tf.nn.embedding_lookup(self.LT, input)
            embedded = tf.nn.dropout(embedded, keep_prob=K)  #[B,Ss,E]
        return embedded

    def blstm(self, input, layer, i, l):
        cfg = layer.split('-')
        hunits = int(cfg[1])
        K = 1.0 - float(cfg[2])

        with tf.variable_scope("blstm_{}".format(i), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(hunits, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=K)
            cell_bw = tf.contrib.rnn.LSTMCell(hunits, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=K)
            (output_src_fw, output_src_bw), (last_src_fw, last_src_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, sequence_length=l, dtype=tf.float32)
            output = tf.concat([output_src_fw, output_src_bw], axis=2) #[B,Ss,layers[i]*2]
            last = tf.concat([last_src_fw.h, last_src_bw.h], axis=1) #[B, layers[-1]*2] (i take h since last_state is a tuple with (c,h))
        return output, last

    def lstm(self, input, layer, i, l, origin=None):
        cfg = layer.split('-')
        hunits = int(cfg[1])
        K = 1.0 - float(cfg[2])

        initial_state = None
        if origin is not None:
            with tf.variable_scope("bridge_{}".format(i),reuse=tf.AUTO_REUSE):
                initial_state_h = tf.layers.dense(origin, hunits, use_bias=False) # H*2 or E => H
                initial_state_c = tf.zeros(tf.shape(initial_state_h))
                initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_c, initial_state_h)

        with tf.variable_scope("lstm_{}".format(i), reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.LSTMCell(hunits)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=K)
            output, last = tf.nn.dynamic_rnn(cell, input, initial_state=initial_state, sequence_length=l, dtype=tf.float32)
        return output, last

    def conv(self, input, layer, i):
        ### input are the embedded source words [B,Ss,Es]
        cfg = layer.split('-')
        filters = int(cfg[1])
        kernel_size = int(cfg[2])
        K = 1.0 - float(cfg[3])
        with tf.variable_scope("conv_{}".format(i), reuse=tf.AUTO_REUSE):
            output = tf.layers.conv1d(inputs=input, filters=filters, kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
            output = tf.nn.dropout(output, keep_prob=K)
        return output

    def sembedding(self, out_src, last_src, layer, l):
        if layer == 'last':
            if tf.size(last_src) == 0: 
                sys.stderr.write("error: -net_snt 'last' cannot be used with the last -net_enc layer\n")
                sys.exit()
            return last_src #[B,Hs[-1]*2]

        elif layer == 'max':
            mask = tf.expand_dims(tf.sequence_mask(l, dtype=tf.float32), 2)
            embed_snt = out_src * mask + (1-mask) * tf.float32.min #masked tokens contain -Inf
            embed_snt = tf.reduce_max(embed_snt, axis=1) #[B,H*2] or [B,E] if word embedding
            return embed_snt

        elif layer == 'mean':
            mask = tf.expand_dims(tf.sequence_mask(l, dtype=tf.float32), 2) #[B,S] => [B,S,1]
            embed_snt = out_src * mask #masked tokens contain 0.0
            embed_snt = tf.reduce_sum(embed_snt, axis=1) / tf.expand_dims(tf.to_float(l), 1) #[B,H*2] or [B,E] if word embedding
            return embed_snt

        sys.stderr.write("error: bad -net_snt option '{}'\n".format(layer))
        sys.exit()

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

    def add_encoder_src(self):
        B = tf.shape(self.input_src)[0] #batch size
        S = tf.shape(self.input_src)[1] #seq_length (including <pad> tokens)

        self.embed_src = self.wembedding(self.input_src, self.config.vocab.length, self.config.net_wrd) #[B,S,E]
        self.out_src = self.embed_src
        self.last_src = []
        for i,layer in enumerate(self.config.net_enc.split(',')):
            if layer.startswith('b'): 
                self.out_src, self.last_src = self.blstm(self.out_src, layer, i, self.len_src)
            elif layer.startswith('c'):
                self.out_src = self.conv(self.out_src, layer, i)
            elif layer.startswith('l'):
                self.out_src, self.last_src = self.lstm(self.out_src, layer, i, self.len_src)
            elif layer.startswith('g'):
                self.out_src, self.last_src = self.gru(self.out_src, layer, i, self.len_src)

        self.embed_snt_src = self.sembedding(self.out_src, self.last_src, self.config.net_snt, self.len_src)


    def add_encoder_tgt(self):
        B = tf.shape(self.input_tgt)[0] #batch size
        S = tf.shape(self.input_tgt)[1] #seq_length (including <pad> tokens)

        self.embed_tgt = self.wembedding(self.input_tgt, self.config.vocab.length, self.config.wrd) #[B,S,E]
        self.out_tgt = self.embed_tgt
        self.last_tgt = []
        for i,layer in enumerate(self.config.net_enc.split(',')):
            if layer.startswith('b'): 
                self.out_tgt, self.last_tgt = self.blstm(self.out_tgt, layer, i, self.len_tgt)
            elif layer.startswith('c'):
                self.out_tgt = self.conv(self.out_tgt, layer, i)
            elif layer.startswith('l'):
                self.out_tgt, self.last_tgt = self.lstm(self.out_tgt, layer, i, self.len_tgt)
            elif layer.startswith('g'):
                self.out_tgt, self.last_tgt = self.gru(self.out_tgt, layer, i, self.len_tgt)

        self.embed_snt_tgt = self.sembedding(self.out_tgt, self.last_tgt, self.config.net_snt, self.len_tgt)

#        pars = sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())
#        sys.stderr.write("Total Enc parameters: {} => {}\n".format(pars, GetHumanReadable(pars*4))) #one parameter is 4 bytes (float32)
#        for var in tf.trainable_variables(): 
#            pars = var.get_shape().num_elements()
#            sys.stderr.write("\t{} => {} {}\n".format(pars, GetHumanReadable(pars*4), var))

    def add_decoder(self):
        B = tf.shape(self.input_tgt)[0] #batch size
        S = tf.shape(self.input_tgt)[1] #seq_length

        self.embed_tgt = self.wembedding(self.input_tgt, self.config.vocab.length, self.config.net_wrd)

        #self.embed_snt_src is either [B,H] or [B,E] if only embedding is used
        self.embed_snt_src_extend = tf.expand_dims(self.embed_snt_src, 1) #[B,H] or [B,E] => [B,1,H] or [B,1,E]
        self.embed_snt_src_extend = tf.tile(self.embed_snt_src_extend, [1, S, 1]) #[B,S,H] or [B,S,E]
        self.embed_snt_src_plus_tgt = tf.concat([self.embed_snt_src_extend, self.embed_tgt], 2) #[B,S,H+E] or [B,S,E+E]

        self.out_tgt = self.embed_snt_src_plus_tgt
        for i,layer in enumerate(self.config.net_dec.split(',')):
            if layer.startswith('l'):
                self.out_tgt, _ = self.lstm(self.out_tgt, layer, i, self.len_tgt, origin=self.embed_snt_src)

        with tf.variable_scope("logits",reuse=tf.AUTO_REUSE):
            self.out_logits = tf.layers.dense(self.out_tgt, self.config.vocab.length)
            self.out_pred = tf.argmax(self.out_logits, 2)

        pars = sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())
        sys.stderr.write("Total Enc/Dec parameters: {} => {}\n".format(pars, GetHumanReadable(pars*4))) #one parameter is 4 bytes (float32)
        for var in tf.trainable_variables(): 
            pars = var.get_shape().num_elements()
            sys.stderr.write("\t{} => {} {}\n".format(pars, GetHumanReadable(pars*4), var))

    def add_loss(self):
        Vt = self.config.vocab.length #tgt vocab

        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_ref, depth=self.config.vocab.length, dtype=tf.float32), logits=self.out_logits) #[B, S]
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
        self.add_encoder_src()  
        if self.config.src_tst: ###inference
            if self.config.tgt_tst: ###bitext
                self.add_encoder_tgt()
        else: #training
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
        score = Score()
        pscore = Score() ### partial score
        ini_time = time.time()
        tpre = time.time()
        for iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch) in enumerate(train):
#            if iter==5: self.debug(fd, src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch)
            fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch, lr)
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            score.add(loss,[],[],[])
            pscore.add(loss,[],[],[])
            if (iter+1)%self.config.reports == 0:
                tnow = time.time()
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                sys.stderr.write('{} Epoch {} Iteration {}/{} (loss={:.6f}) lr={:.6f} time={:.2f} sec/iter\n'.format(curr_time,curr_epoch,iter+1,nbatches,pscore.Loss,lr,(tnow-tpre)/self.config.reports))
                tpre = tnow
                pscore = Score()
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        end_time = time.time()
        self.config.tloss = score.Loss
        self.config.time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write('{} Epoch {} TRAIN (loss={:.4f}) time={:.2f} sec'.format(curr_time,curr_epoch,score.Loss,end_time-ini_time))
        sys.stderr.write(' Train set: words={}/{} %oov={:.2f}/{:.2f}\n'.format(train.nsrc_tok, train.ntgt_tok, 100.0*train.nsrc_unk/train.nsrc_tok, 100.0*train.ntgt_unk/train.ntgt_tok))
        #keep records
        self.config.seconds = "{:.2f}".format(end_time - ini_time)
        self.config.last_epoch += 1
        self.save_session(self.config.last_epoch)

        ##########################
        # evaluate over devset ###
        ##########################
        score = Score()
        if dev is not None:
            nbatches = (dev.len + self.config.batch_size - 1) // self.config.batch_size
            ini_time = time.time()
            for iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, nsrc_unk_batch, ntgt_unk_batch, len_src_batch, len_tgt_batch) in enumerate(dev):
                fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch)
                loss, out_pred, tmask = self.sess.run([self.loss,self.out_pred,self.tmask], feed_dict=fd)
                score.add(loss,out_pred,ref_batch,tmask)
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            end_time = time.time()
            sys.stderr.write('{} Epoch {} VALID (loss={:.4f} Acc={:.2f}) time={:.2f} sec'.format(curr_time,curr_epoch,score.Loss,score.Acc,end_time-ini_time))
            sys.stderr.write(' Valid set: words={}/{} %oov={:.2f}/{:.2f}\n'.format(dev.nsrc_tok, dev.ntgt_tok, 100.0*dev.nsrc_unk/dev.nsrc_tok, 100.0*dev.ntgt_unk/dev.ntgt_tok))
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

            fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch)
#            self.debug2(fd,src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch)
            if tst.is_bitext:
                embed_snt_src_batch, embed_snt_tgt_batch = self.sess.run([self.embed_snt_src, self.embed_snt_tgt], feed_dict=fd)
            else:
                embed_snt_src_batch = self.sess.run(self.embed_snt_src, feed_dict=fd)

            for i_sent in range(len(embed_snt_src_batch)):
                result = []
                ### normalization
                embed_snt_src_batch[i_sent] = embed_snt_src_batch[i_sent] / np.linalg.norm(embed_snt_src_batch[i_sent])
                if tst.is_bitext:
                    embed_snt_tgt_batch[i_sent] = embed_snt_tgt_batch[i_sent] / np.linalg.norm(embed_snt_tgt_batch[i_sent])

                if self.config.show_sim:
                    if tst.is_bitext: result.append("{:.6f}".format(self.compute_sim(embed_snt_src_batch[i_sent], embed_snt_tgt_batch[i_sent])))

                if self.config.show_oov:
                    result.append("{}".format(nsrc_unk_batch[i_sent]))
                    if tst.is_bitext: result.append("{}".format(ntgt_unk_batch[i_sent]))

                if self.config.show_emb: 
                    result.append(" ".join(["{}".format(e) for e in embed_snt_src_batch[i_sent]]))
                    if tst.is_bitext: result.append(" ".join(["{}".format(e) for e in embed_snt_tgt_batch[i_sent]]))

                if self.config.show_snt: 
                    result.append(" ".join(raw_src_batch[i_sent]))
                    if tst.is_bitext: result.append(" ".join(raw_tgt_batch[i_sent]))

                if self.config.show_idx: 
                    result.append(" ".join([str(e) for e in src_batch[i_sent]]))
                    if tst.is_bitext: result.append(" ".join([str(e) for e in tgt_batch[i_sent]]))

                print "\t".join(result)

        end_time = time.time()
        toks = tst.nsrc_tok
        if tst.is_bitext:
            toks += tst.ntgt_tok
        stoks_per_sec = toks / (end_time - ini_time)
        sents_per_sec = tst.len / (end_time - ini_time)
        sys.stderr.write("Analysed {} sentences with {}/{} tokens in {:.2f} seconds => {:.2f} toks/sec {:.2f} sents/sec (model/testset loading times not considered)\n".format(tst.len, tst.nsrc_tok, tst.ntgt_tok, end_time - ini_time, stoks_per_sec, sents_per_sec))

    def compute_sim(self, src, tgt):
        ### src and tgt are already normalized 
        sim = np.sum(src * tgt) ### cosine
#        sim = np.sqrt(np.sum(np.power(a-b,2) for a, b in zip(src, tgt))) #euclidean
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


#################
### other #######
#################

    def debug(self, fd, src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch):
        embed_src, out_src, last_src, initial_state, embed_snt_src, embed_tgt, embed_snt_src_plus_tgt, out_tgt, out_logits, out_pred = self.sess.run([self.embed_src, self.out_src, self.last_src, self.initial_state, self.embed_snt_src, self.embed_tgt, self.embed_snt_src_plus_tgt, self.out_tgt, self.out_logits, self.out_pred], feed_dict=fd)
        sys.stderr.write("Encoder\n")
        sys.stderr.write("B={}\n".format(len(src_batch)))
        sys.stderr.write("Ss={}\n".format(len(src_batch[0])))
        sys.stderr.write("Es={}\n".format(self.config.net_wrd_len))
        if (len(self.config.net_blstm_lens)): sys.stderr.write("Hs[-1]={}\n".format(self.config.net_blstm_lens[-1]/2))
        sys.stderr.write("shape of embed_src = {} [B,Ss,Es]\n".format(np.array(embed_src).shape))
        sys.stderr.write("shape of out_src = {} [B,Ss,Hs[-1]*2] or [B,Ss,Es]\n".format(np.array(out_src).shape))
        sys.stderr.write("shape of last_src = {} [B,Hs[-1]*2]\n".format(np.array(last_src).shape))
        sys.stderr.write("shape of embed_snt_src = {}\n".format(np.array(embed_snt_src).shape))
        sys.stderr.write("Decoder\n")
        sys.stderr.write("Ht={}\n".format(self.config.net_lstm_len))
        sys.stderr.write("St={}\n".format(len(tgt_batch[0])))
        sys.stderr.write("Et={}\n".format(self.config.net_wrd_len))
        sys.stderr.write("Vt={}\n".format(self.config.vocab.length))
        sys.stderr.write("shape of initial_state = {} [B,Ht]\n".format(np.array(initial_state).shape))
        sys.stderr.write("shape of embed_tgt = {} [B,St,Et]\n".format(np.array(embed_tgt).shape))
        sys.stderr.write("shape of embed_snt_src_plus_tgt = {} [B,St,Hs[-1]*2+Et] or [B,St,Es+Et]\n".format(np.array(embed_snt_src_plus_tgt).shape))
        sys.stderr.write("shape of out_tgt = {} [B,St,Ht]\n".format(np.array(out_tgt).shape))
        sys.stderr.write("shape of out_logits = {} [B,St,Vt]\n".format(np.array(out_logits).shape))
        sys.stderr.write("shape of out_pred = {} [B,St]\n{}\n".format(np.array(out_pred).shape, out_pred))
#        print("src[0]"," ".join([str(e) for e in raw_src_batch[0]]))
#        print("isrc[0]"," ".join([str(e) for e in src_batch[0]]))
#        print("len_src[0]",str(len_src_batch[0]))
#        print("tgt[0]"," ".join([str(e) for e in raw_tgt_batch[0]]))
#        print("itgt[0]"," ".join([str(e) for e in tgt_batch[0]]))
#        print("iref[0]"," ".join([str(e) for e in ref_batch[0]]))
#        print("len_tgt[0]",str(len_tgt_batch[0]))

    def debug2(self, fd, src_batch, len_src_batch, tgt_batch, len_tgt_batch, ref_batch):
        sys.stderr.write("src_batch {}\n".format(src_batch))
        sys.stderr.write("len_src_batch {}\n".format(len_src_batch))
        sys.stderr.write("tgt_batch {}\n".format(tgt_batch))
        sys.stderr.write("len_tgt_batch {}\n".format(len_tgt_batch))
        sys.stderr.write("ref_batch {}\n".format(ref_batch))
        embed_src, out_src, embed_snt_src = self.sess.run([self.embed_src, self.out_src, self.embed_snt_src], feed_dict=fd)
        sys.stderr.write("shape of embed_src = {}\n".format(np.array(embed_src).shape))
        sys.stderr.write("shape of out_src = {}\n".format(np.array(out_src).shape))
        sys.stderr.write("shape of embed_snt_src = {}\n".format(np.array(embed_snt_src).shape))



