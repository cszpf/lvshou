# encoding=utf-8
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import sys
import time
sys.path.append("../project")
from divide import load_data
from project.interface import SupperModel
import pickle

PATH = "../../data/Content"
TEST_PATH = "../../data/Sample"


class Word2vec(SupperModel):
    def __init__(self, alone, size):
        self.alone = alone
        self.data = None
        self.vocabulary = None
        self.model = None
        self.token = "transData.sentenceList"
        self.counter = None
        self.size = size
        super(SupperModel, self).__init__()

    def load_corpus(self):
        rules = os.listdir(PATH)
        rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
        if not self.alone:
            suffix = "_agent_tokens.csv"
        else:
            suffix = "_tokens.csv"
        data = pd.DataFrame()
        for rule in rules:
            _ = load_data(os.path.join(PATH, rule, rule + suffix))
            data = pd.concat([data, _], axis=0)

        # 样本空间
        data.drop_duplicates(['UUID'], inplace=True)
        data.reset_index(inplace=True)
        print("All corpus size:", len(data))
        self.data = data

    def get_vocabulary(self, min_df, max_df, max_features):
        pickle_file = "w2v_counter.pkl"
        if os.path.exists(os.path.join(TEST_PATH, pickle_file)):
            print("load counter_vectorizer...")
            self.counter = pickle.load(open(os.path.join(TEST_PATH, 'w2v', pickle_file), 'rb'))
        else:
            print("fitting in data: ", self.data[self.token].shape)
            self.counter = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
            self.counter.fit(self.data[self.token])
            pickle.dump(self.counter, open(os.path.join(TEST_PATH, 'w2v', pickle_file), 'wb'))
        self.vocabulary = [word for word, num in self.counter.vocabulary_.items()]
        print(self.vocabulary)
        print(len(self.vocabulary))

    def train_model(self):
        model = Word2Vec(self.data[self.token], size=self.size, window=5, min_count=3, workers=4)
        save_path = os.path.join(TEST_PATH, 'w2v', "w2v_model.model")
        model.save(save_path)

    def vector(self):
        print('载入模型中')
        model = Word2Vec.load(os.path.join(TEST_PATH, 'w2v', "w2v_model.model"))
        print('加载成功')
        vec = np.zeros((len(self.data), self.size))

        for i in range(len(self.data)):
            print(i)
            counter = 0
            for word in self.data[self.token][i].split(' '):
                try:
                    vec[i] += np.array(model[word])
                    counter += 1
                except Exception as error:
                    pass
                    # print(error)
            if counter != 0:
                vec[i] = vec[i] / float(counter)  # 求均值
        np.save(os.path.join(TEST_PATH, 'w2v', "vec.npy"), vec)
        self.data['UUID'].to_csv(os.path.join(TEST_PATH, 'w2v', "uuid_all.txt"), sep=',',
                                 encoding="utf-8", index=False)

    def filter(self):
        for i in range(len(self.data)):
            print(i)
            words = []
            for word in self.data[self.token][i].split(' '):
                if word in self.vocabulary:
                    words.append(word)
            self.data[self.token][i] = ' '.join(words)
        self.data[['UUID', self.token]].to_csv(os.path.join(TEST_PATH, 'w2v', "w2v_sentences.csv"), sep=',',
                                               encoding="utf-8", index=False)


if __name__ == "__main__":
    start_time = time.time()
    w2v = Word2vec(alone=True, size=300)
    w2v.load_corpus()
    # w2v.train_model()
    w2v.vector()
    # w2v.get_vocabulary(min_df=3, max_df=0.5, max_features=80000)
    # w2v.filter()
    print('time cost is', time.time() - start_time)
