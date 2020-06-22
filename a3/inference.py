import os
from gensim.models import Word2Vec
import re
import sys

def main(data_path):
    with open(os.path.join(data_path, 'sample.txt')) as f:
        sample_text = f.readlines()

    lines =[]
    for s in sample_text:
        sentence = s.strip().split()
        lines.append(sentence)

    for words in lines:
        for w in range(len(words)):
            test = re.sub(r'[^a-zA-Z0-9,.\']', " ", str(words[w]))
            words[w] = test
    w2v = Word2Vec.load("./data/w2v.model")

    print('Good: {}'.format( w2v.wv.most_similar(positive=["good"], topn=20)))
    print(" ")
    print('Bad: {}'.format(w2v.wv.most_similar(positive=["bad"], topn=20)))
    return

if __name__ == '__main__':
    main(sys.argv[1])