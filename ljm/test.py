import pandas as pd
import numpy as np
import sys
import time
import pickle
sys.path.append("..")
import os
from project.divide import load_data, PATH1, PATH2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


if __name__ == "__main__":
    PATH = "../../data/Content"
    rule1 = "过度承诺效果"
    rule2 = "违反1+1模式"
    test_file = "sample1"
    pickle_file = "Vectorizer_total_ngram_1_2.pkl"
    n = 5
    # counter1 = pickle.load(open(os.path.join(PATH, rule1, test_file[:-1], pickle_file), 'rb'))
    # counter2 = pickle.load(open(os.path.join(PATH, rule2, test_file[:-1], pickle_file), 'rb'))
    # print(sorted(counter1.vocabulary_))
    # print(sorted(counter2.vocabulary_))
    l = [1, 2, 3, 4, 5]
    index = [0, 2, 4]
    print(np.array(l)[index])
