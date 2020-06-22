import os
import sys
import random
import re
from gensim.models import Word2Vec

def read_dataset(data_path):
    """
    reads the raw dataset and returns all the lines as a list of string
    """
    # os.path can be used for seamless path construction across different
    # operating systems.

    with open(os.path.join(data_path, 'pos.txt')) as f:
        pos_lines = f.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        neg_lines = f.readlines()
    all_lines = pos_lines + neg_lines
    return list(zip(all_lines, ["pos"]*len(pos_lines) + ["neg"]*len(neg_lines)))
def main(data_path):
    """
    Train a word2vec model on the given dataset
    """
    # Read raw data
    all_lines = read_dataset(data_path)
    random.shuffle(all_lines)


    lines = []
    label = []
    #Split each sentence in the list, and append to result list
    for s in all_lines:
        sentence = s[0].strip().split()
        l = s[1]
        # construct the entry to be added to the csv files
        lines.append(sentence)
        label.append(l)

    for words in lines:
        for w in range(len(words)):
            test = re.sub(r'[^a-zA-Z0-9,.\']', " ", str(words[w]))
            words[w]=test



    print('Training word2vec model')
    # This will take some to finish
    w2v = Word2Vec( lines, size=100, window=5, min_count=1, workers=4)
    w2v.save('a3/data/w2v.model')



if __name__ == '__main__':
    main(sys.argv[1])