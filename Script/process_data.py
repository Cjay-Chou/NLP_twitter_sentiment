import os
import re
import pandas as pd
import numpy as np
import keras
import pickle

def data_cleaning(data):
    #change to lower char
    data['SentimentText'] = data['SentimentText'].str.lower()
    #make sure size
    data['SentimentText'].fillna(value="nan", inplace=True)

    #use RE remove none use mark
    f1 = lambda a: re.sub(r'(@.*? )', '', a)
    f2 = lambda a: re.sub(r'(@.*?$)', '', a)
    f3 = lambda a: re.sub(' +', ' ', a)
    data['SentimentText'] = data['SentimentText'].apply(f1)
    data['SentimentText'] = data['SentimentText'].apply(f2)
    data['SentimentText'] = data['SentimentText'].apply(f3)

    return data

def flatten(l):
    return [item for sublist in l for item in sublist]

def preprocessing(data_x):
    print("start preprocessing")
    raw_text = data_x['SentimentText']
    tok_raw = keras.preprocessing.text.Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    data_x['seq_sentimenttext'] = tok_raw.texts_to_sequences(data_x['SentimentText'])
    words_num = np.unique(flatten(data_x['seq_sentimenttext'])).shape[0] + 1
    print('unique words number:',words_num)

    data_X = keras.preprocessing.sequence.pad_sequences(data_x['seq_sentimenttext'], maxlen=MAX_LEN)
    return data_X, words_num

def get_data(data_path):
    data = pd.read_csv(data_path, sep=",", error_bad_lines=False)
    print('data nums: ',pd.value_counts(data['Sentiment']))
    data = data_cleaning(data)

    data_X, words_num = preprocessing(data)
    data_Y = keras.utils.to_categorical(data['Sentiment'], num_classes=2)

    data = {'X':data_X,'Y':data_Y,'words_num':words_num}
    return data

if __name__ == '__main__':
    MAX_LEN = 50
    DATA_PATH = '../data/train.csv'
    data = get_data(DATA_PATH)

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(data, f)
