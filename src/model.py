# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import sys
import os
import time
from random import randint
from config import Config
from idf import SentIdf

def print2D(t,m):
    sys.stderr.write(t+"\n")
    for row in m:
        sys.stderr.write(" ".join(["{:10.3f}".format(e) for e in row]))
        sys.stderr.write("\n")

def print1D(t,m):
    sys.stderr.write(t+"\n")
    sys.stderr.write(" ".join(["{:10.3f}".format(e) for e in m]))
    sys.stderr.write("\n")

def print0D(t,m):
    sys.stderr.write(t+"\n")
    sys.stderr.write("{:10.3f}\n".format(m))

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
        self.Loss = self.sumloss / (1.0 * self.total_iters)
        if len(tout):
            masked_equals = np.logical_and(np.equal(tout,tref), tmask)
            self.total_words += np.sum(tmask)
            self.sumok += np.count_nonzero(masked_equals)
            self.Acc = 100.0 * self.sumok / self.total_words

class Model():
    def __init__(self, config):
        self.config = config
        self.sess = None


    def wembedding(self, input, V, E, dropout, namelayer):
        K = 1.0 - dropout
        if self.config.is_inference: K = 1.0
        sys.stderr.write("\twembedding V={} E={} K={:.3f} name={}\n".format(V,E,K,namelayer))

        with tf.device('/cpu:0'), tf.variable_scope("embedding_{}".format(namelayer), reuse=tf.AUTO_REUSE): ### same embeddings for src/tgt words
            self.LT = tf.get_variable(initializer = tf.random_uniform([V, E], minval=-0.1, maxval=0.1), dtype=tf.float32, name="LT")
            embedded = tf.nn.embedding_lookup(self.LT, input)
            embedded = tf.nn.dropout(embedded, keep_prob=K)  #[B,Ss,E]
        return embedded


    def blstm(self, input, hunits, dropout, seq_length, namelayer):
        K = 1.0 - dropout
        if self.config.is_inference: K = 1.0
        sys.stderr.write("\tblsmt hunits={} K={:.3f} name={}\n".format(hunits,K,namelayer))

        with tf.variable_scope("blstm_{}".format(namelayer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(hunits, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
#            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=K)
            cell_bw = tf.contrib.rnn.LSTMCell(hunits, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed), state_is_tuple=True)
#            cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=K)
            (output_src_fw, output_src_bw), (last_src_fw, last_src_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, sequence_length=seq_length, dtype=tf.float32)
            output = tf.concat([output_src_fw, output_src_bw], axis=2) #[B,Ss,layers[i]*2]
            output = tf.nn.dropout(output, keep_prob=K)
            last = tf.concat([last_src_fw.h, last_src_bw.h], axis=1) #[B, layers[-1]*2] (i take h since last_state is a tuple with (c,h))
            last = tf.nn.dropout(last, keep_prob=K)
        return output, last


    def lstm(self, input, hunits, dropout, seq_length, namelayer, origin=None):
        K = 1.0 - dropout
        if self.config.is_inference: K = 1.0
        sys.stderr.write("\tlsmt hunits={} K={:.3f} bridge={} name={}\n".format(hunits,K,origin==None,namelayer))

        initial_state = None
        if origin is not None:
            with tf.variable_scope("bridge_{}".format(namelayer),reuse=tf.AUTO_REUSE):
                initial_state_h = tf.layers.dense(origin, hunits, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed)) # H*2 or E => H
                initial_state_c = tf.zeros(tf.shape(initial_state_h))
                initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_c, initial_state_h)

        with tf.variable_scope("lstm_{}".format(namelayer), reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.LSTMCell(hunits, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=self.config.seed))
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=K)
            output, last = tf.nn.dynamic_rnn(cell, input, initial_state=initial_state, sequence_length=seq_length, dtype=tf.float32)
        return output, last


    def conv(self, input, filters, kernel_size, dropout, namelayer):
        ### input are the embedded source words [B,Ss,Es]
        K = 1.0 - dropout
        if self.config.is_inference: K = 1.0
        sys.stderr.write("\tconv filters={} kernel_size={} K={:.3f} name={}\n".format(filters,kernel_size,K,namelayer))

        with tf.variable_scope("conv_{}".format(namelayer), reuse=tf.AUTO_REUSE):
            output = tf.layers.conv1d(inputs=input, filters=filters, kernel_size=kernel_size, padding="same", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(-0.01, 0.01, seed=self.config.seed))
            output = tf.nn.dropout(output, keep_prob=K)
        return output


    def sembedding(self, out, last, layer, l, tfidf=None):
        sys.stderr.write("\tsembedding {}\n".format(layer))

        if layer == 'last':
            if tf.size(last) == 0: 
                sys.stderr.write("error: sentence embedding 'last' cannot be used with the previous layer\n")
                sys.exit()
            return last #[B,Hs[-1]*2]

        elif layer == 'max':
            if tfidf is not None:
                tfidf = tf.expand_dims(tfidf, 2) #[B,S] => [B,S,1]
                out = out * tfidf
            mask = tf.expand_dims(tf.sequence_mask(l, dtype=tf.float32), 2) #[B,S] => [B,S,1]
            embed_snt = out * mask + (1-mask) * tf.float32.min #masked tokens contain -Inf
            embed_snt = tf.reduce_max(embed_snt, axis=1) #[B,H*2] or [B,E] if word embedding
            return embed_snt

        elif layer == 'mean':
            if tfidf is not None:
                tfidf = tf.expand_dims(tfidf, 2) #[B,S] => [B,S,1]
                out = out * tfidf
            mask = tf.expand_dims(tf.sequence_mask(l, dtype=tf.float32), 2) #[B,S] => [B,S,1]
            embed_snt = out * mask #masked tokens contain 0.0
            embed_snt = tf.reduce_sum(embed_snt, axis=1) / tf.expand_dims(tf.to_float(l), 1) #[B,H*2] or [B,E] if word embedding
            return embed_snt

        sys.stderr.write("error: bad sentence embedding option '{}'\n".format(layer))
        sys.exit()

###################
### build graph ###
###################
    
    def add_placeholders(self):
        self.lr            = tf.placeholder(tf.float32, shape=[],          name="lr")
        self.input_src     = tf.placeholder(tf.int32,   shape=[None,None], name="input_src")  # Shape: batch_size x |Fj| (sequence length)
        self.input_tgt     = tf.placeholder(tf.int32,   shape=[None,None], name="input_tgt")  # Shape: batch_size x |Ei| (sequence length)
        self.len_src       = tf.placeholder(tf.int32,   shape=[None],      name="len_src")
        self.len_tgt       = tf.placeholder(tf.int32,   shape=[None],      name="len_tgt")
        self.wrd_tgt       = tf.placeholder(tf.int32,   shape=[None,None], name="wrd_tgt")  # Shape: batch_size x |Ei| (sequence length)
        self.ref_tgt       = tf.placeholder(tf.int32,   shape=[None,None], name="ref_tgt")  # Shape: batch_size x |Ei| (sequence length)
        self.len_wrd       = tf.placeholder(tf.int32,   shape=[None],      name="len_wrd")
        self.div_src       = tf.placeholder(tf.float32, shape=[None,None], name="div_src")  # Shape: batch_size x |Ei| (sequence length)
        self.div_tgt       = tf.placeholder(tf.float32, shape=[None,None], name="div_tgt")  # Shape: batch_size x |Ei| (sequence length)

    def add_encoder(self, side):
        sys.stderr.write("encoder_{}\n".format(side))
        if side=='src':
            B = tf.shape(self.input_src)[0] #batch size
            S = tf.shape(self.input_src)[1] #seq_length (including <pad> tokens)
            Output = self.input_src
            Length = self.len_src
            Vocab = self.config.vocab_src
        elif side=='tgt':
            B = tf.shape(self.input_tgt)[0] #batch size
            S = tf.shape(self.input_tgt)[1] #seq_length (including <pad> tokens)
            Output = self.input_tgt
            Length = self.len_tgt
            Vocab = self.config.vocab_tgt
        else:
            sys.stderr.write("error: bad encoder side={}\n".format(side))
            sys.exit()

        Last = []
        Embed = []
        for l in range(self.config.network.nlayers('enc')):
            t, f, u, d, n = self.config.network.layer('enc',l) ### type, filters, units, dropout, name
            if t=='w':
                Output = self.wembedding(Output, Vocab.length, u, d, n) #[B,S,E]
            elif t=='b': 
                Output, Last = self.blstm(Output, u, d, Length, n)
            elif t=='c':
                Output = self.conv(Output, f, u, d, n)
            elif t=='l':
                Output, Last = self.lstm(Output, u, d, Length, n)
            elif t=='s':
                Embed = self.sembedding(Output, Last, n, Length)
            else:
                sys.stderr.write("error: bad encoder {}-th layer type={}\n".format(l,t))
                sys.exit()

        return Output, Embed


    def add_align(self):
        sys.stderr.write("align\n")
        B = tf.shape(self.input_tgt)[0] #batch size
        Ss = tf.shape(self.input_src)[1] #seq_length
        St = tf.shape(self.input_tgt)[1] #seq_length
        R = 1.0
        Output = self.input_tgt
        Length = self.len_tgt
        Vocab = self.config.vocab_tgt
        Last = []
        Embed = []
        for l in range(self.config.network.nlayers('ali')):
            t, f, u, d, n = self.config.network.layer('ali',l) ### type, filters, units, dropout, name
            if t=='w':
                Output = self.wembedding(Output, Vocab.length, u, d, n) #[B,S,E]
            elif t=='b': 
                Output, Last = self.blstm(Output, u, d, Length, n)
            elif t=='c':
                Output = self.conv(Output, f, u, d, n)
            elif t=='l':
                Output, Last = self.lstm(Output, u, d, Length, n)
            elif t=='s':
                Embed = self.sembedding(Output, Last, n, Length) #not used
            else:
                sys.stderr.write("error: bad ali {}-th layer type={}\n".format(l,t))
                sys.exit()

        self.out_tgt = Output
        self.embed_snt_tgt = Embed

        with tf.name_scope("align"):
            self.align = tf.map_fn(lambda (x, y): tf.matmul(x, tf.transpose(y)), (self.out_src, self.out_tgt), dtype=tf.float32, name="align") #[B,Ss,St]
            self.align_t = tf.transpose(self.align, [0, 2, 1]) #[B,St,Ss]

            #equation (2) [aggregation of each src word]
            self.exp_rs_src = tf.exp(R * self.align_t) #[B,St,Ss]
            self.sum_exp_rs_src = tf.map_fn(lambda (x, l): tf.reduce_sum(x[1:l-1, :], 0), (self.exp_rs_src, self.len_tgt), dtype=tf.float32) #[B,Ss] (do not sum over <bos> and <eos> [1:l-1])
            self.log_sum_exp_rs_src = tf.log(self.sum_exp_rs_src) #[B,Ss]
            self.aggr_src = tf.divide(self.log_sum_exp_rs_src, R, name="aggregation_src") #[B,Ss]
            ### equation (3) [error of each src word]
            self.aggr_times_div_src = self.aggr_src * self.div_src
            self.error_src = tf.log(1 + tf.exp(self.aggr_times_div_src)) ### low error if aggregation_src * div_src is negative (different sign)

            #equation (2) [aggregation of each tgt word]
            self.exp_rs_tgt = tf.exp(R * self.align) #[B,Ss,St]
            self.sum_exp_rs_tgt = tf.map_fn(lambda (x, l): tf.reduce_sum(x[1:l-1, :], 0), (self.exp_rs_tgt, self.len_src), dtype=tf.float32) #[B,St] (do not sum over <bos> and <eos>[1,l-1])
            self.log_sum_exp_rs_tgt = tf.log(self.sum_exp_rs_tgt) #[B,St]
            self.aggr_tgt = tf.divide(self.log_sum_exp_rs_tgt, R, name="aggregation_tgt") #[B,St]
            ### equation (3) [error of each tgt word]
            self.aggr_times_div_tgt = self.aggr_tgt * self.div_tgt
            self.error_tgt = tf.log(1 + tf.exp(self.aggr_times_div_tgt))

            self.sum_error_src = tf.map_fn(lambda (x, l): tf.reduce_sum(x[1:l-1]), (self.error_src, self.len_src), dtype=tf.float32)
            self.loss_src = tf.reduce_mean(self.sum_error_src)

            self.sum_error_tgt = tf.map_fn(lambda (x, l): tf.reduce_sum(x[1:l-1]), (self.error_tgt, self.len_tgt), dtype=tf.float32)
            self.loss_tgt = tf.reduce_mean(self.sum_error_tgt)

            self.aloss = self.loss_tgt + self.loss_src

    def add_translate(self):
        sys.stderr.write("translate\n")
        B = tf.shape(self.input_tgt)[0] #batch size
        S = tf.shape(self.wrd_tgt)[1] #seq_length

        t, f, u, d, n = self.config.network.layer('trn',0) 
        if t=='w':
            self.embed_tgt = self.wembedding(self.wrd_tgt, self.config.vocab_tgt.length, u, d, n) #[B,S,E]
        else:
            sys.stderr.write("error: first tgt layer must be of type w:wembedding instead of '{}'\n".format(t))
            sys.exit()

        #self.embed_snt_src is either [B,H] or [B,E] if only embedding is used
        self.embed_snt_src_extend = tf.expand_dims(self.embed_snt_src, 1) #[B,H] or [B,E] => [B,1,H] or [B,1,E]
        self.embed_snt_src_extend = tf.tile(self.embed_snt_src_extend, [1, S, 1]) #[B,S,H] or [B,S,E]
        self.embed_snt_src_plus_tgt = tf.concat([self.embed_snt_src_extend, self.embed_tgt], 2) #[B,S,H+E] or [B,S,E+E]
        self.out_tgt = self.embed_snt_src_plus_tgt

        t, f, u, d, n = self.config.network.layer('trn',1) 
        if t=='l':
            self.out_tgt, _ = self.lstm(self.out_tgt, u, d, self.len_wrd, n, origin=self.embed_snt_src)
        elif t=='b':
            self.out_tgt, _ = self.blstm(self.out_tgt, u, d, self.len_wrd, n, origin=self.embed_snt_src)
        else:
            sys.stderr.write("error: second tgt layer must be of type l:lstm or b:blstm instead of '{}'\n".format(t))
            sys.exit()

        with tf.variable_scope("logits",reuse=tf.AUTO_REUSE):
            self.out_logits = tf.layers.dense(self.out_tgt, self.config.vocab_tgt.length)
            self.out_pred = tf.argmax(self.out_logits, 2)

        with tf.name_scope("trans"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.ref_tgt, depth=self.config.vocab_tgt.length, dtype=tf.float32), logits=self.out_logits) #[B, S]
            self.tmask = tf.sequence_mask(self.len_wrd, dtype=tf.float32) #[B, S]            
            self.tloss = tf.reduce_sum(xentropy*self.tmask) / tf.to_float(tf.reduce_sum(self.len_wrd))


    def add_direct(self):
        sys.stderr.write("direct\n")
        B = tf.shape(self.input_tgt)[0] #batch size
        Ss = tf.shape(self.input_src)[1] #seq_length
        St = tf.shape(self.input_tgt)[1] #seq_length

        with tf.name_scope("direct"):
            self.embed_snt_prod = tf.abs(tf.multiply(self.embed_snt_src, self.embed_snt_tgt)) # eq 4
            self.embed_snt_diff = tf.subtract(self.embed_snt_src, self.embed_snt_tgt) # eq 5
            self.hi = tf.concat([self.embed_snt_diff, self.embed_snt_prod], 2)

            for l in range(self.config.network.nlayers('dir')):
                if self.config.network.layer('dir',l)>0:
                    self.hi = tf.layers.dense(self.hi, self.config.network.layer('dir',l), activation=tf.tanh)

            self.snt_ref = tf.reduce_mean(-1.0*self.div_src,1) #-1.0:divergent or 1.0:same 
            self.p = tf.layers.dense(self.hi, 1, activation=tf.sigmoid)
            self.snt_ref = tf.max(tf.reduce_mean(-1.0 * self.div_src,1), 0.0) #1.0 or 0.0 (same or divergent)
            self.dloss = tf.reduce_sum(self.snt_ref*log(self.p) + (1-self.snt_ref)*log(1-self.p))


    def add_train(self):
        if   self.config.network.opt == 'adam':     self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.config.network.opt == 'sgd':      self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.config.network.opt == 'adagrad':  self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.config.network.opt == 'rmsprop':  self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.config.network.opt == 'adadelta': self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        else:
            sys.stderr.write("error: bad -net_opt option '{}'\n".format(self.config.network.opt))
            sys.exit()

        if self.config.clip > 0.0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.clip)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.train_op = self.optimizer.minimize(self.loss)


    def build_graph(self):
        self.add_placeholders()

        self.out_src, self.embed_snt_src = self.add_encoder('src')

        if self.config.is_inference and self.cofig.src_tst is not None:
            self.out_tgt, self.embed_snt_tgt = self.add_encoder('tgt')

        else: ### training
            self.loss = 0
            if self.config.network.dir is not None:
                self.add_direct()  
                self.loss += self.dloss

            if self.config.network.trn is not None:
                self.add_translate()  
                self.loss += self.tloss

            if self.config.network.ali is not None:
                self.add_align()
                self.loss += self.aloss

            self.add_train()

        pars = sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())
        sys.stderr.write("Total Enc/Dec parameters: {} => {}\n".format(pars, GetHumanReadable(pars*4))) #one parameter is 4 bytes (float32)
        for var in tf.trainable_variables(): 
            pars = var.get_shape().num_elements()
            sys.stderr.write("\t{} => {} {}\n".format(pars, GetHumanReadable(pars*4), var))

###################
### feed_dict #####
###################

    def get_feed_dict(self, src, len_src, tgt=[[]], len_tgt=[], wrd_tgt=[[]], ref_tgt=[[]], len_wrd=[[]], div_src=[[]], div_tgt=[[]], lr=0.0):
        feed = { 
            self.input_src: src,
            self.input_tgt: tgt,
            self.len_src: len_src,
            self.len_tgt: len_tgt,
            self.wrd_tgt: wrd_tgt,
            self.ref_tgt: ref_tgt,
            self.len_wrd: len_wrd,
            self.div_src: div_src,
            self.div_tgt: div_tgt,
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
        do_debug = False
        for iter, (src_batch, tgt_batch, wrd_batch, ref_batch, div_src_batch, div_tgt_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch, len_wrd_batch) in enumerate(train):
            fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, wrd_batch, ref_batch, len_wrd_batch, div_src_batch, div_tgt_batch, lr)
            if iter%10000==0: 
                do_debug = True
            if do_debug:
                do_debug = self.debug(iter, fd, src_batch, tgt_batch, wrd_batch, ref_batch, div_src_batch, div_tgt_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch, len_wrd_batch)
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            score.add(loss,[],[],[])
            pscore.add(loss,[],[],[])
            if iter%self.config.reports == 0:
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
            for iter, (src_batch, tgt_batch, wrd_batch, ref_batch, div_src_batch, div_tgt_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch, len_wrd_batch) in enumerate(dev):
                fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch, wrd_batch, ref_batch, len_wrd_batch, div_src_batch, div_tgt_batch, lr)
                loss = self.sess.run(self.loss, feed_dict=fd)
                score.add(loss,[],[],[])
#                loss, out_pred, tmask = self.sess.run([self.loss,self.out_pred,self.tmask], feed_dict=fd)
#                score.add(loss,out_pred,ref_tgt_batch,tmask)
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            end_time = time.time()
            sys.stderr.write('{} Epoch {} VALID (loss={:.6f}) time={:.2f} sec'.format(curr_time,curr_epoch,score.Loss,end_time-ini_time))
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
        for iter, (src_batch, tgt_batch, ref_src_batch, ref_tgt_batch, div_src_batch, div_tgt_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(tst):

            fd = self.get_feed_dict(src_batch, len_src_batch, tgt_batch, len_tgt_batch)
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
                    nsrc_unk = sum(1 for idx in src_batch[i_sent] if idx==self.config.vocab_src.idx_unk)
                    result.append("{}".format(nsrc_unk))
                    if tst.is_bitext: 
                        ntgt_unk = sum(1 for idx in tgt_batch[i_sent] if idx==self.config.vocab_tgt.idx_unk)
                        result.append("{}".format(ntgt_unk))

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
                sys.stderr.write("Initializing model variables\n")
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

    def debug(self, it, fd, src_batch, tgt_batch, wrd_batch, ref_batch, div_src_batch, div_tgt_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch, len_wrd_batch):
        if len(src_batch[0]) > 15 or len(tgt_batch[0]) > 15:
            return True #do debug again
        sys.stderr.write("iter={}\n".format(it))
        sys.stderr.write("B={}\n".format(len(src_batch)))
        sys.stderr.write("Ss={}\n".format(len(src_batch[0])))
        sys.stderr.write("St={}\n".format(len(tgt_batch[0])))
        sys.stderr.write("Vs={}\n".format(self.config.vocab_src.length))
        sys.stderr.write("Vt={}\n".format(self.config.vocab_tgt.length))
        if self.config.network.ali is not None:
            out_src, embed_snt_src, \
            out_tgt, embed_snt_tgt, \
            align_t, exp_rs_src, sum_exp_rs_src, log_sum_exp_rs_src, aggr_src, aggr_times_div_src, error_src, sum_error_src, loss_src, \
            align,   exp_rs_tgt, sum_exp_rs_tgt, log_sum_exp_rs_tgt, aggr_tgt, aggr_times_div_tgt, error_tgt, sum_error_tgt, loss_tgt, \
            loss = self.sess.run([\
                self.out_src, self.embed_snt_src, \
                self.out_tgt, self.embed_snt_tgt, \
                self.align_t, self.exp_rs_src, self.sum_exp_rs_src, self.log_sum_exp_rs_src, self.aggr_src, self.aggr_times_div_src, self.error_src, self.sum_error_src, self.loss_src, \
                self.align,   self.exp_rs_tgt, self.sum_exp_rs_tgt, self.log_sum_exp_rs_tgt, self.aggr_tgt, self.aggr_times_div_tgt, self.error_tgt, self.sum_error_tgt, self.loss_tgt, \
                self.loss], feed_dict=fd)

            sys.stderr.write("Encoder src\n")
            sys.stderr.write("shape of out_src = {} [B,Ss,Hs] or [B,Ss,Es]\n".format(np.array(out_src).shape))
            sys.stderr.write("shape of embed_snt_src = {}\n".format(np.array(embed_snt_src).shape))
            sys.stderr.write("Encoder tgt\n")
            sys.stderr.write("shape of out_tgt = {} [B,St,Ht] or [B,St,Et]\n".format(np.array(out_tgt).shape))
            sys.stderr.write("shape of embed_snt_tgt = {}\n".format(np.array(embed_snt_tgt).shape))

            for b in range(len(align)):
                if b==0: # debug only the first example in this batch
                    sys.stderr.write("### {} #######################\n".format(b))
                    sys.stderr.write("src\t{}\n".format(" ".join(str(e) for e in raw_src_batch[b])))
                    sys.stderr.write("isrc\t{}\n".format(" ".join([str(e) for e in src_batch[b]])))
                    sys.stderr.write("tgt\t{}\n".format(" ".join(str(e) for e in raw_tgt_batch[b])))
                    sys.stderr.write("itgt\t{}\n".format(" ".join([str(e) for e in tgt_batch[b]])))
                    sys.stderr.write("iwrd\t{}\n".format(" ".join([str(e) for e in wrd_batch[b]])))
                    sys.stderr.write("iref\t{}\n".format(" ".join([str(e) for e in ref_batch[b]])))
                    sys.stderr.write("div_src\t{}\n".format(" ".join([str(e) for e in div_src_batch[b]])))
                    sys.stderr.write("div_tgt\t{}\n".format(" ".join([str(e) for e in div_tgt_batch[b]])))
                    sys.stderr.write("lens[src={}, tgt={}, wrd={}]\n".format(len_src_batch[b], len_tgt_batch[b], len_wrd_batch[b]))
                    #print2D("src out[{}]".format(b), out_src[b])
                    print1D("src out[{}][1]".format(b), out_src[b][1])
                    print2D("src Align_t[{}]".format(b), align_t[b])
                    print2D("exp(A)[{}]".format(b), exp_rs_src[b])
                    print1D("sum_exp(A)[{}]".format(b), sum_exp_rs_src[b])
                    print1D("log_sum_exp(A)[{}]".format(b), log_sum_exp_rs_src[b])
                    print1D("aggr[{}]".format(b), aggr_src[b])
                    print1D("aggr_times_div[{}]".format(b), aggr_times_div_src[b])
                    print1D("error[{}]".format(b), error_src[b])
                    print0D("sum_error_src[{}]".format(b),sum_error_src[b])    
                    #print2D("tgt out[{}]".format(b), out_tgt[b])
                    print1D("tgt out[{}][1]".format(b), out_tgt[b][1])
                    print2D("tgt Align[{}]".format(b), align[b])
                    print2D("exp(A)[{}]".format(b), exp_rs_tgt[b])
                    print1D("sum_exp(A)[{}]".format(b), sum_exp_rs_tgt[b])
                    print1D("log_sum_exp(A)[{}]".format(b), log_sum_exp_rs_tgt[b])
                    print1D("aggr[{}]".format(b), aggr_tgt[b])
                    print1D("aggr_times_div[{}]".format(b), aggr_times_div_tgt[b])
                    print1D("error[{}]".format(b), error_tgt[b])
                    print0D("sum_error_tgt[{}]".format(b),sum_error_tgt[b])

                print0D("sum_error_src[{}]".format(b),sum_error_src[b])    
                print0D("sum_error_tgt[{}]".format(b),sum_error_tgt[b])
    
            sys.stderr.write("#############################\n")
            print0D("loss_src (mean)",loss_src)
            print0D("loss_tgt (mean)",loss_tgt)
            print0D("loss",loss)  
            return False #do not debug again
            #sys.exit()




