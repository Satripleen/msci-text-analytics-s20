import os
import sys
from pprint import pprint
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    x_train_1 = read_csv(os.path.join(data_dir, 'train_1.csv'))
    x_val_1 = read_csv(os.path.join(data_dir, 'val_1.csv'))
    x_test_1 = read_csv(os.path.join(data_dir, 'test_1.csv'))
    labels = read_csv(os.path.join(data_dir, 'label.csv'))
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train) + len(x_val)]
    y_test = labels[-len(x_test):]
    y_train_1 = labels[:len(x_train_1)]
    y_val_1 = labels[len(x_train_1): len(x_train_1) + len(x_val_1)]
    y_test_1 = labels[-len(x_test_1):]
    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_1, x_val_1, x_test_1, y_train_1, y_val_1, y_test_1


def train_for_unigrams_with_stopwords(x_train, y_train):
    print('Calling CountVectorizer for unigrams with stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 1))
    x_train_count = count_vect.fit_transform(x_train)
    print('Building Tf-idf vectors for unigrams with stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB with for unigrams with stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train)
    filename = "./data/mnb_uni.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    return clf, count_vect, tfidf_transformer

def train_for_unigrams_without_stopwords(x_train_1, y_train_1):
    print('Calling CountVectorizer for unigram without stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 1))
    x_train_count = count_vect.fit_transform(x_train_1)
    print('Building Tf-idf vectors for unigram without stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB for unigrams without stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train_1)
    filename = "./data/mnb_uni_ns.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    return clf, count_vect, tfidf_transformer

def train_for_bigrams_with_stopwords(x_train, y_train):
    print('Calling CountVectorizer for bigrams with stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (2, 2))
    x_train_count = count_vect.fit_transform(x_train)
    print('Building Tf-idf vectors for bigrams with stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB for bigrams with stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train)
    filename1 = "./data/mnb_bi.pkl"
    pickle.dump(clf, open(filename1, 'wb'))
    return clf, count_vect, tfidf_transformer



def train_for_bigrams_without_stopwords(x_train_1, y_train_1):
    print('Calling CountVectorizer for bigrams without stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (2, 2))
    x_train_count = count_vect.fit_transform(x_train_1)
    print('Building Tf-idf vectors for bigrams without stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB for bigrams without stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train_1)
    filename = "./data/mnb_bi_ns.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    return clf, count_vect, tfidf_transformer

def train_for_unigram_bigram_with_stopwords(x_train, y_train):
    print('Calling CountVectorizer for unigram_bigrams with stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 2))
    x_train_count = count_vect.fit_transform(x_train)
    print('Building Tf-idf vectors for unigram_bigrams with stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB for unigram_bigrams with stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train)
    filename = "./data/mnb_uni_bi.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    return clf, count_vect, tfidf_transformer

def train_for_unigram_bigram_without_stopwords(x_train_1, y_train_1):
    print('Calling CountVectorizer for unigram_bigram without stopwords')
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 2))
    x_train_count = count_vect.fit_transform(x_train_1)
    print('Building Tf-idf vectors for unigram_bigram without stopwords')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    print('Training MNB for unigram_bigram without stopwords')
    clf = MultinomialNB(alpha=0.75).fit(x_train_tfidf, y_train_1)
    filename = "./data/mnb_uni_bi_ns.pkl"
    pickle.dump(clf, open(filename, 'wb'))
    return clf, count_vect, tfidf_transformer



def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds,average='macro'),
        'recall': recall_score(y, preds,average='macro'),
        'f1': f1_score(y, preds,average= 'macro',labels=np.unique(preds)),
        }


def main(data_dir):
    """
    loads the dataset along with labels, trains a simple MNB classifier
    and returns validation and test scores in a dictionary
    """
    # load data
    x_train, x_val, x_test, y_train, y_val, y_test, x_train_1, x_val_1, x_test_1, y_train_1, y_val_1, y_test_1= load_data(data_dir)
    # train without stopwords for unigrams
    clf, count_vect, tfidf_transformer = train_for_unigrams_with_stopwords(x_train, y_train)

    scores_1 = {}
    # validate
    print('Validating for unigram with stopwords')
    scores_1['val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for unigram with stopwords')
    scores_1['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    # train with stopwords for unigrams
    clf, count_vect, tfidf_transformer = train_for_unigrams_without_stopwords(x_train_1, y_train_1)

    scores_2 = {}
    # validate
    print('Validating for unigram without stopwords')
    scores_2['val'] = evaluate(x_val_1, y_val_1, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for unigram without stopwords')
    scores_2['test'] = evaluate(x_test_1, y_test_1, clf, count_vect, tfidf_transformer)


# train without stopwords for bigrams
    clf, count_vect, tfidf_transformer = train_for_bigrams_with_stopwords(x_train, y_train)
    scores_3 = {}
    # validate
    print('Validating for bigrams with stopwords')
    scores_3['val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for bigrams with stopwords')
    scores_3['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    # train with stopwords for bigrams
    clf, count_vect, tfidf_transformer = train_for_bigrams_without_stopwords(x_train_1, y_train_1)


    scores_4 = {}
    # validate
    print('Validating for bigrams without stopwords')
    scores_4['val'] = evaluate(x_val_1, y_val_1, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for bigrams without stopwords')
    scores_4['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)


# train without stopwords for unigrams and bigrams
    clf, count_vect, tfidf_transformer = train_for_unigram_bigram_with_stopwords(x_train, y_train)
    scores_5 = {}
    # validate
    print('Validating for unigram and bigrams with stopwords')
    scores_5['val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for unigrams and bigrams with stopwords')
    scores_5['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    # train with stopwords for unigrams and bigrams
    clf, count_vect, tfidf_transformer = train_for_unigram_bigram_without_stopwords(x_train_1, y_train_1)


    scores_6 = {}
    # validate
    print('Validating for unigrams and bigrams without stopwords')
    scores_6['val'] = evaluate(x_val_1, y_val_1, clf, count_vect, tfidf_transformer)
    # test
    print('Testing for unigrams and bigrams without stopwords')
    scores_6['test'] = evaluate(x_test_1, y_test_1, clf, count_vect, tfidf_transformer)
    return scores_1,scores_2,scores_3,scores_4,scores_5,scores_6

if __name__ == '__main__':
    pprint(main(sys.argv[1]))