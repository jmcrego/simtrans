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

    if len(config.src_trn) and len(config.tgt_trn) and len(config.tgt_trn_lid) and len(config.src_val) and len(config.tgt_val) and len(config.tgt_val_lid):
        trn = Dataset(config.src_trn, config.tgt_trn, config.tgt_trn_lid, config)
        val = Dataset(config.src_val, config.tgt_val, config.tgt_val_lid, config)
        model.learn(trn, val, config.n_epochs)

    elif config.src_tst:
        if config.tgt_tst is None: 
            tst = Dataset([config.src_tst], [], [], config)
        else:
            tst = Dataset([config.src_tst], [config.tgt_tst], [], config)
        model.inference(tst)

    model.close_session()

if __name__ == "__main__":
    main()
