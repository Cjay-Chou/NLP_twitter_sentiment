import os
import numpy as np
import pandas as pd
import keras
from keras import layers as klayer
import keras.backend as K

args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build NLP Net program')
    parser.add_argument("outfile", help="Output model structure file in YAML format (*.yml).")
    parser.add_argument("-p", "--patchsize", help="Patch size. (ex. 44x44x28)", default="44x44x28")
    parser.add_argument("-c", "--nclasses", help="Number of classes of segmentaiton including background.", default=2, type=int)
    parser.add_argument("-r", "--reduction", help="The number of filters for filter reduction.", default=128, type=int)
    parser.add_argument("--noreduction", help="Do not use filter reduction.", dest="use_reduction", action="store_false")
    parser.add_argument("--nobn", help="Do not use batch normalization layer", dest="use_bn", action='store_false')
    parser.add_argument("--nodropout", help="Do not use dropout layer", dest="use_dropout", action='store_false')
    parser.add_argument("-v", "--unetversion", help="Unet version.", choices=["v1", "v2", "v3"], default="v1")
    #parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args

N_CLASSES=self.n_classes, MAX_TEXT=self.MAX_TEXT, MAX_ITEM_DESC_SEQ=self.MAX_ITEM_DESC_SEQ
def build_net():
    sentimenttext = Input(shape=[X_train.shape[1]], name="seq_sentimenttext")



if __name__ == '__main__':
    args = ParseArgs()
    build_net()