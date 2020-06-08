import os
import sys
import argparse
import re
import random

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
    print("read files")
    return all_lines
def main(data_path):
    """
    reads the raw dataset from data_path, creates a vocab dictionary,
    "tokenizes" the sentences in the dataset.
    NOTE: In our case tokenization simply refers to splitting a sentence at ' '
    """

    # Read raw data
    all_lines = read_dataset(data_path)
    #all_lines = all_lines.split("\n")

    lines = []
    #Split each sentence in the list, and append to result list
    for s in all_lines:
        lines.append(s.split())

    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"," "]

    ## Removing the special charachetrs
    for words in lines:
        for w in range(len(words)):
            test = re.sub(r'[^a-zA-Z0-9,.\']', " ", str(words[w]))
            words[w]=test
    print("Special caharcters removed ")

    random.shuffle(lines)

    train_set = lines[:int(0.8*len(lines))]
    val_set = lines[int(0.8*len(lines)):int(0.9*len(lines))]
    test_set = lines[int(0.9*len(lines)):]

    with open("train.csv","w") as f:
        for i in train_set:
                for j in i:
                    f.write(j + ",")
                f.write("\n")

    with open("val.csv","w") as f:
        for i in val_set:
            for j in i:
                f.write(j + ",")
            f.write("\n")

    with open ("test.csv" , "w") as f:
        for i in test_set:
            for j in i:
                f.write(j +",")
            f.write("\n")


    with open("out.csv_1","w") as f:
        for i in lines:
            for j in i:
                f.write(j + ",")
            f.write("\n")
    print("csv created")
    temp_list_1=[]
    for main_itr in range(len(lines)):
        for j in range(len(lines[main_itr])):
            if (lines[main_itr][j]).lower() not in stop_words:
                temp_list_1.append(lines[main_itr][j])
        lines[main_itr]=temp_list_1.copy()
        temp_list_1.clear()
    print("stopwords removed")
    train_set_1 = lines[:int(0.8*len(lines))]
    val_set_1 = lines[int(0.8*len(lines)):int(0.9*len(lines))]
    test_set_1 = lines[int(0.9*len(lines)):]

    with open("train_1.csv","w") as f:
        for i in train_set_1:
            for j in i:
                f.write(j + ",")
            f.write("\n")

    with open("val_1.csv","w") as f:
        for i in val_set_1:
            for j in i:
                f.write(j + ",")
            f.write("\n")

    with open ("test_1.csv" , "w") as f:
        for i in test_set_1:
            for j in i:
                f.write(j +",")
            f.write("\n")



    with open("out.csv_2","w") as f:
        for i in lines:
            for j in i:
                f.write(j + ",")
            f.write("\n")
    print("csv created")

if __name__ == "__main__":
    input = sys.argv[1]


