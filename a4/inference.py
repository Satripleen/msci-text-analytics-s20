"""
Inference script to test reconstruction quality of the trained autoencoder
"""
import sys
import os
from gensim.models import Word2Vec
from keras.models import load_model
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, Dropout
# from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing import text, sequence
import numpy as np
from keras.utils import to_categorical
from keras.regularizers import l2


def load_data(data_dir):
    with open(os.path.join(data_dir, 'test.csv')) as f:
        x_test = f.readlines()
        x_test = [[(word[0:-1]).lower() for word in sen.strip('\n[]').split(', ')] for sen in x_test]
        x_test = [' '.join(line) for line in x_test]
    return x_test

def target(data_dir):
    with open(os.path.join(data_dir, 'label.csv')) as f:
        label = f.readlines()
    label = [0 if w.strip('\n')=='neg' else 1 for w in label]
    train_data_size = int(len(label)*0.8)
    val_data_size = int(len(label)*0.1)
    train_label = label[:train_data_size]
    val_label = label[train_data_size:train_data_size+val_data_size]
    test_label = label[train_data_size+val_data_size:]
    test_label = to_categorical(np.asarray(test_label))
    return (np.array(test_label))

def embedding_mat(w2v, test_data):
    vocab_size_max = 20000
    embedding_dim = 100
    max_doc_len = 26
    t = text.Tokenizer( num_words= vocab_size_max,filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',lower = False, oov_token= 1)

    t.fit_on_texts(test_data)
    x_test = t.texts_to_sequences(test_data)
    x_test = pad_sequences(x_test, maxlen=max_doc_len, truncating='post', padding='post')
    vocab_size = min(len(t.word_index), vocab_size_max)
    vocab_size = len(t.word_index) + 1
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in t.word_index.items():
        embedding_vector = None
        if (i == 1):
            embedding_vector = np.random.randn(1, embedding_dim)
        else:
            try:
                embedding_vector = w2v[word]
            except:
                continue
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    return embedding_matrix, x_test

def main (data_dir,model_dir):
    x_test = load_data(data_dir)

    test_label = target(data_dir)

    # x_test = pad_sequences(x_train, maxlen=max_doc_len, truncating='post', padding='post')

    w2v = Word2Vec.load("../a3/data/w2v.model")

    embedding_matrix, x_test = embedding_mat(w2v, x_test)
    cnn = load_model(model_dir)
    # x_test=x_test[0:100]
    # test_label=test_label[0:100]
    predict = cnn.predict_classes(x_test)
    predict =np.array(predict)
    print('predictions',predict)

    loss,accuracy = cnn.evaluate(x_test,test_label)
    print("loss is : ",loss," accuracy : ",accuracy*100)

    return loss,accuracy

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])




