import sys
import os
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import  Embedding, Dense, Dropout,Flatten
from keras.preprocessing import text
import numpy as np
from keras.regularizers import l2
from keras.utils import to_categorical


def load_data(data_dir):
    with open(os.path.join(data_dir, 'train.csv')) as f:
        x_train = f.readlines()
        x_train = [[(word[0:-1]) for word in sen.strip('\n[]').split(', ')] for sen in x_train]
        x_train = [' '.join(line) for line in x_train]
    with open(os.path.join(data_dir, 'val.csv')) as f:
        x_val = f.readlines()
        x_val = [[(word[0:-1]) for word in sen.strip('\n[]').split(', ')] for sen in x_val]
        x_val = [' '.join(line) for line in x_val]
    with open(os.path.join(data_dir, 'test.csv')) as f:
        x_test = f.readlines()
        x_test = [[(word[0:-1]) for word in sen.strip('\n[]').split(', ')] for sen in x_test]
        x_test = [' '.join(line) for line in x_test]
    return x_train,x_val,x_test

def target(data_dir):
    with open(os.path.join(data_dir, 'label.csv')) as f:
        label = f.readlines()
    label = [0 if w.strip('\n')=='neg' else 1 for w in label]
    train_data_size = int(len(label)*0.8)
    val_data_size = int(len(label)*0.1)
    train_label = label[:train_data_size]
    train_label = to_categorical(np.asarray(train_label))
    val_label = label[train_data_size:train_data_size+val_data_size]
    val_label = to_categorical(np.asarray(val_label))
    test_label = label[train_data_size+val_data_size:]
    test_label = to_categorical(np.asarray(test_label))
    return(np.array(train_label),np.array(val_label),np.array(test_label))

def embedding_mat(w2v, train_data, val_data):
    vocab_size_max = 20000
    embedding_dim = 100
    max_doc_len = 26
    # t = text.Tokenizer(num_words=vocab_size_max, filters='', lower=False, split=' ', oov_token=1)
    t = text.Tokenizer( num_words= vocab_size_max,filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',lower = False, oov_token= 1)

    t.fit_on_texts(train_data)
    x_train = t.texts_to_sequences(train_data)
    x_val = t.texts_to_sequences(val_data)
    x_train = pad_sequences(x_train, maxlen=max_doc_len, truncating='post', padding='post')
    x_val = pad_sequences(x_val, maxlen=max_doc_len, truncating='post', padding='post')
    print(x_train[1])
    print(len(t.word_index))
    # vocab_size = min(len(t.word_index), vocab_size_max)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    input_len=len(t.word_index)+1
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
    return embedding_matrix, x_train, x_val

def main(data_dir):
    x_train,x_val,x_test = load_data(data_dir)
    train_label,val_label,test_label = target(data_dir)

    w2v = Word2Vec.load("../a3/data/w2v.model")

    embedding_matrix, x_train, x_val = embedding_mat(w2v, x_train, x_val)

    max_doc_len = 26
    model = Sequential()
    x=(1,26)

    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=max_doc_len, trainable=False, weights=[embedding_matrix]))
    print(model.summary())

    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.1), name='output_layer'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())
    # fit the model
    model.fit(x_train, train_label, epochs=20, validation_data=(x_val, val_label), batch_size=512)
    model.save("./data/nn_tanh.model")

    """ The model with sigmoid and relu are commented below for which the test accuracy is best"""

    # model.add(
    #     Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=max_doc_len,
    #               trainable=False, weights=[embedding_matrix]))
    # print(model.summary())
    # # model.add(LSTM(units = 512,activation='tanh', kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
    #
    # model.add(Dense(64, activation='sigmoid', kernel_regularizer=l2(0.001)))
    # # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    #
    #
    # model.add(Flatten())
    # model.add(Dropout(0.4))
    # model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.1), name='output_layer'))
    # # model.add(Flatten())
    # # model.add(Dropout(0.4))
    # # model.add(Dense(2, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # # fit the model
    # model.fit(x_train, train_label, epochs=20, validation_data=(x_val, val_label), batch_size=4096)
    # model.save("./data/nn_sigmoid.model")

    #model.add(
    #     Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=max_doc_len,
    #               trainable=False, weights=[embedding_matrix]))
    # print(model.summary())
    # # model.add(LSTM(units = 512,activation='tanh', kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
    #

    # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    #
    #
    # model.add(Flatten())
    # model.add(Dropout(0.4))
    # model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.1), name='output_layer'))
    # # model.add(Flatten())
    # # model.add(Dropout(0.4))
    # # model.add(Dense(2, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # # fit the model
    # model.fit(x_train, train_label, epochs=20, validation_data=(x_val, val_label), batch_size=4096)
    # model.save("./data/nn_relu.model")

if __name__ == '__main__':
    main(sys.argv[1])

