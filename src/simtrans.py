# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys
from dataset import Dataset, Vocab
from model import Model
from config import Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    config = Config(sys.argv)
    model = Model(config)
    model.build_graph()
    model.initialize_session()

    if config.src_trn and config.tgt_trn and config.lid_trn and config.src_val and config.tgt_val and config.lid_val:
        trn = Dataset(config.src_trn,config.tgt_trn,config.lid_trn, config, do_shuffle=True)
        val = Dataset(config.src_val,config.tgt_val,config.lid_val, config, do_shuffle=False)
        model.learn(trn, val, config.n_epochs)

    elif config.src_tst:
        tst = Dataset(config.src_tst,config.tgt_tst,None, config, do_shuffle=False)
        model.inference(tst)

    model.close_session()

if __name__ == "__main__":
    main()
