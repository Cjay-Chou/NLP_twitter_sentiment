# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import keras
import argparse
import yaml
from keras import layers as klayers
import keras.backend as K
import pickle


from attention_context import AttentionWithContext

args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build NLP Net program')
    parser.add_argument("infile", help="input training file in pickl format (*.pkl).")
    parser.add_argument("outfile", help="Output model structure file in HDF5 format (*.h5).")
    parser.add_argument("-c", "--nclasses", help="Number of classes of segmentaiton including background.", default=2, type=int)
    parser.add_argument("--nencoder", help="number of encoder units", default = 16, type=int)
    parser.add_argument("--ndecoder", help="number of decoder units", default = 16, type=int)
    #parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args


def build_branch_cnn(emb):
    """
    like googlenet inception v2 structure, after simplified
    """
    m_cnn_1 = klayers.Conv1D(64, 3, padding='same',activation='relu')(emb)
    m_cnn_2 = klayers.Conv1D(32, 3, padding='same',activation='relu')(emb)
    m_cnn_2 = klayers.Conv1D(128, 3, padding='same',activation='relu')(m_cnn_2)
    m_cnn_3 = klayers.Conv1D(128, 3, padding='same',activation='relu')(emb)
    m_cnn_3 = klayers.Conv1D(64, 3, padding='same',activation='relu')(m_cnn_3)
    m_cnn = klayers.concatenate([m_cnn_1, m_cnn_2, m_cnn_3])
    m_cnn = klayers.MaxPooling1D(pool_size=2, padding='valid')(m_cnn)
    m_cnn = klayers.Conv1D(64, 3, padding='same',activation='relu')(m_cnn)
    m_cnn = klayers.MaxPooling1D(pool_size=2, padding='valid')(m_cnn)
    m_cnn = klayers.Flatten()(m_cnn)
    return m_cnn

def build_branch_bilstm_am(emb):
    """
    adding attention model
    """
    # m_lstm = LSTM(self.encoder_units, return_sequences=True, trainable=True)(emb)
    m_lstm = klayers.Bidirectional(klayers.LSTM(args.nencoder, return_sequences=True, trainable=True))(emb)
    attention = AttentionWithContext()(m_lstm)

    return attention

def build_branch_bilstm_position_am(emb):
    """
    build attention model according position infomation
    result not good.
    """
    m_lstm = klayers.Bidirectional(klayers.LSTM(args.nencoder, return_sequences=True, trainable=True))(emb)
    attention = klayers.TimeDistributed(klayers.Dense(1, activation='tanh'))(m_lstm)
    attention = klayers.Flatten()(attention)
    attention = klayers.Activation('softmax')(attention)
    attention = klayers.RepeatVector(args.ndecoder * 2)(attention)
    attention = klayers.Permute([2, 1])(attention)

    m_lstm_am = klayers.merge([m_lstm, attention], mode='mul')
    m_lstm_am = klayers.Lambda(lambda xin: K.sum(xin, axis=1))(m_lstm_am)

    return m_lstm_am

def build_net():
    """
    # Build model
    """
    inputs = klayers.Input(shape=[50], name="seq_sentimenttext")
    emb_sentimenttext = klayers.Embedding(MAX_TEXT, 50, trainable=True)(inputs)
    #get two branch
    layer_cnn = build_branch_cnn(emb_sentimenttext)
    layer_lstm_am = build_branch_bilstm_am(emb_sentimenttext)
    #add two branch
    close_branch = klayers.concatenate([layer_cnn, layer_lstm_am])

    fc = klayers.Dense(128, activation='relu')(close_branch)
    fc = klayers.Dropout(0.2)(fc)
    fc = klayers.Dense(64, activation='relu')(fc)
    fc = klayers.Dropout(0.2)(fc)

    output = klayers.Dense(args.nclasses, activation='softmax')(fc)
    model = keras.models.Model(inputs,output)

    print(model.summary())

    return model

if __name__ == '__main__':
    args = ParseArgs()
    
    with open(args.infile, 'rb') as f:
        data_f = pickle.load(f)
    
    MAX_TEXT = data_f['words_num']

    model = build_net()
    if args.outfile is not None:
        model.save(args.outfile)
