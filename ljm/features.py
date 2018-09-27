# encoding=utf-8
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import time
sys.path.append("../project")
from divide import load_data, PATH1, PATH2
import pickle
PATH = "../../data/Content"
TEST_PATH = "../../data/Sample"


def get_window_words(sentence, key_words, windows):
    words = []
    all_words = sentence.split(' ')
    index = [i for i, x in enumerate(all_words) if x in key_words]
    for i in index:
        begin = i - windows
        end = i + windows + 1
        if begin < 0:
            begin = 0
        if end > len(all_words):
            end = len(all_words)
        words.extend(all_words[begin:end])
    # print(len(all_words), len(index), len(words))
    return ' '.join(words)


class Features(object):
    def __init__(self, rule, alone, max_df, min_df, max_features, window=None, use_idf=False):
        self.rule = rule
        self.alone = alone
        self.path = os.path.join(PATH, rule)
        if not self.alone:
            self.data = load_data(os.path.join(PATH, rule, rule + "_agent_tokens.csv"))
        else:
            self.data = load_data(os.path.join(PATH, rule, rule + "_tokens.csv"))
        self.tokens = "transData.sentenceList"
        self.seed = 2018
        self.Counter = None
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.use_idf = use_idf
        self.window = window

    def sample(self, test_uuid, only=True):
        """
        采样训练集数据，首先将出现在测试集中的数据去除
        然后使用剩余数据的所有正样本，采样相同数量的负样本
        将训练集 UUID 保存在 self.path 路径中
        :param test_uuid: 测试集数据 UUID
        :param only: only 为 True，负样本仅从不出现任何违规的数据中提取
        :return: 训练集
        """
        self.data = self.data[~self.data['UUID'].isin(test_uuid.values[:, 0])]
        print("pos data:", len(self.data))
        print("是否只从不出现任何违规的数据中采集负样本: " + str(only))
        # 负样本从不出现任何违规的数据中提取
        if only:
            if not self.alone:
                file_name = os.path.join(PATH, "不违规", "不违规_agent_tokens.csv")
            else:
                file_name = os.path.join(PATH, "不违规", "不违规_tokens.csv")
            neg_data = load_data(file_name)
        else:
            rules = os.listdir(PATH)
            rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
            rules.remove(self.rule)
            if not self.alone:
                suffix = "_agent_tokens.csv"
            else:
                suffix = "_tokens.csv"
            neg_data = pd.DataFrame()
            for rule in rules:
                _ = load_data(os.path.join(PATH, rule, rule + suffix))
                neg_data = pd.concat([neg_data, _], axis=0)

        # 负样本空间
        neg_data.drop_duplicates(['UUID'], inplace=True)
        neg_data = neg_data[~neg_data['UUID'].isin(test_uuid.values[:, 0])]
        neg_data = neg_data[~neg_data['UUID'].isin(self.data['UUID'])]
        print("neg data:", len(neg_data))

        train_data = pd.concat([self.data, neg_data], axis=0)
        # train_data = pd.concat([self.data, neg_data.sample(n=len(self.data) * 2, random_state=self.seed)], axis=0)
        train_data = train_data.sample(frac=1, random_state=self.seed)
        return train_data

    def load_train(self, test_file, only):
        """
        采样训练集数据，首先将出现在测试集中的数据去除
        然后使用剩余数据的所有正样本，采样相同数量的负样本
        将训练集 UUID 保存在 self.path 路径中
        :param test_file: 测试集UUID文件
        :param only: only 为 True，负样本仅从不出现任何违规的数据中提取
        """
        # 已经提取过训练集，直接加载返回
        if only:
            file_name = self.rule + "_train_only_" + test_file + ".csv"
        else:
            file_name = self.rule + "_train_" + test_file + ".csv"
        if os.path.exists(os.path.join(self.path, test_file[:-1], file_name)):
            self.data = load_data(os.path.join(self.path, test_file[:-1], file_name))
            return

        print("sample train data...")
        test_uuid = pd.read_csv(os.path.join('../../data/Sample', test_file + ".txt"), header=None)
        self.data = self.sample(test_uuid, only=only)
        self.data.reset_index(drop=True, inplace=True)
        if self.window:
            print("window:", self.window)
            key_words = []
            with open(os.path.join('../setting', self.rule + ".txt"), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    key_words.append(line.strip())
            self.data[self.tokens] = self.data[self.tokens].apply(lambda x: get_window_words(x, key_words,
                                                                                             windows=self.window))
        print(len(self.data))
        # self.data.to_csv(os.path.join(self.path, test_file[:-1], file_name), sep=',', encoding="utf-8", index=False)

    def load_test(self, test_file):
        test_uuid = pd.read_csv(os.path.join('../../data/Sample', test_file + ".txt"), header=None)
        rules = os.listdir(PATH)
        rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
        if not self.alone:
            suffix = "_agent_tokens.csv"
        else:
            suffix = "_tokens.csv"
        test_data = pd.DataFrame()
        for rule in rules:
            _ = load_data(os.path.join(PATH, rule, rule + suffix))
            test_data = pd.concat([test_data, _], axis=0)

        # 测试集样本空间
        test_data.drop_duplicates(['UUID'], inplace=True)
        test_data.reset_index(inplace=True)
        print(len(test_data))
        self.data = test_data[test_data['UUID'].isin(test_uuid.values[:, 0])]
        self.data.reset_index(drop=True, inplace=True)
        if self.window:
            print("window:", self.window)
            key_words = []
            with open(os.path.join('../setting', self.rule + ".txt"), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    key_words.append(line.strip())
            self.data[self.tokens] = self.data[self.tokens].apply(lambda x: get_window_words(x, key_words,
                                                                                             windows=self.window))

    def get_label(self, _file):
        label = []
        # 对每个数据样本，遍历其检测出的违规类型
        for counter in range(len(self.data)):
            if self.rule not in self.data['analysisData.illegalHitData.ruleNameList'][counter]:
                label.append(0)
            else:
                for i, item in enumerate(self.data['analysisData.illegalHitData.ruleNameList'][counter]):
                    if self.rule == item:
                        label.append(1 if self.data['correctInfoData.correctResult'][counter].
                                     get("correctResult")[i] == '1' else 0)
        # for i in range(len(label)):
        #     if label[i] == 1:
        #         print(self.data['analysisData.illegalHitData.ruleNameList'][i],
        #               self.data['correctInfoData.correctResult'][i])
        np.array(label).dump(_file)

    def get_weight(self, test_file, only, total=False, train=True):
        # if total and not os.path.exists(os.path.join(self.path, test_file[:-1], "Vectorizer_total_ngram_1_2.pkl")):
        if total and not os.path.exists(os.path.join("../../data", "Vectorizer_total_ngram_1_2.pkl")):
            print("generate vocabulary...")
            rules = os.listdir(PATH)
            rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
            if not self.alone:
                suffix = "_agent_tokens.csv"
            else:
                suffix = "_tokens.csv"
            total_data = pd.DataFrame()
            for rule in rules:
                _ = load_data(os.path.join(PATH, rule, rule + suffix))
                total_data = pd.concat([total_data, _], axis=0)

            # 样本空间
            total_data.drop_duplicates(['UUID'], inplace=True)
            total_data.reset_index(inplace=True)

            if self.window:
                print("window:", self.window)
                key_words = []
                with open(os.path.join('../setting', self.rule + ".txt"), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        key_words.append(line.strip())
                    total_data[self.tokens] = total_data[self.tokens].apply(lambda x: get_window_words(x, key_words,
                                                                            windows=self.window))
            print("fitting in data: ", total_data.shape)
            self.Counter = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, use_idf=True,
                                           max_features=self.max_features, ngram_range=(1, 2))
            # self.Counter = CountVectorizer(max_df=self.max_df, min_df=self.min_df,
            #                                max_features=self.max_features)
            self.Counter.fit(total_data[self.tokens])
            if not os.path.exists(os.path.join(self.path, test_file[:-1])):
                os.mkdir(os.path.join(self.path, test_file[:-1]))
            if not os.path.exists(os.path.join(self.path, 'sample_proportion')):
                os.mkdir(os.path.join(self.path, test_file[:-1]))
            pickle.dump(self.Counter, open(os.path.join(self.path, test_file[:-1], "Vectorizer_total_ngram_1_2.pkl"), 'wb'))
            pickle.dump(self.Counter,
                        open(os.path.join(self.path, 'sample_proportion', "Vectorizer_total_ngram_1_2.pkl"), 'wb'))

        if train:
            if not os.path.exists(os.path.join(self.path, test_file[:-1])):
                os.makedirs(os.path.join(self.path, test_file[:-1]))
            print("load train data...")
            # 生成保存文件名
            if only:
                file_name = self.rule + "_train_weight_only_" + test_file + ".pkl"
                label_name = self.rule + "_train_label_only_" + test_file + ".npy"
                if not total:
                    pickle_file = "CountVectorizer_" + test_file + "_only" + ".pkl"
                else:
                    pickle_file = "Vectorizer_total_ngram_1_2.pkl"
            else:
                file_name = self.rule + "_train_weight_" + test_file + ".pkl"
                label_name = self.rule + "_train_label_" + test_file + ".npy"
                if not total:
                    pickle_file = "CountVectorizer_" + test_file + ".pkl"
                else:
                    pickle_file = "Vectorizer_total_ngram_1_2.pkl"

            # 特征文件不存在，生成
            # ../../data/Sample/rule/sample/label_name
            if not os.path.exists(os.path.join(self.path, test_file[:-1], label_name)):
                self.load_train(test_file, only)

                # if os.path.exists(os.path.join(self.path, test_file[:-1], pickle_file)):
                if os.path.exists(os.path.join("../../data", pickle_file)):
                    print("load counter_vectorizer...")
                    self.Counter = pickle.load(open(os.path.join("../../data", pickle_file), 'rb'))
                    # self.Counter = pickle.load(open(os.path.join(self.path, test_file[:-1], pickle_file), 'rb'))
                else:
                    print("fitting in data: ", self.data.shape)
                    self.Counter = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, use_idf=True,
                                                   max_features=self.max_features, ngram_range=(1, 1))
                    # self.Counter = CountVectorizer(max_df=self.max_df, min_df=self.min_df,
                    #                                max_features=self.max_features)
                    self.Counter.fit(self.data[self.tokens])
                    pickle.dump(self.Counter, open(os.path.join(self.path, test_file[:-1], pickle_file), 'wb'))

                print("get label...")
                self.get_label(os.path.join(self.path, test_file[:-1], label_name))
                print("get weight...")
                token_counter = self.Counter.transform(self.data['transData.sentenceList'].values)
                print(len(self.Counter.vocabulary_.items()))
                weight = token_counter.toarray()
                print(weight.shape)
                pickle.dump(token_counter, open(os.path.join(self.path, test_file[:-1], file_name), 'wb'))

        # 测试集特征
        else:
            print("load test data...")
            # 生成保存文件名
            if only:
                file_name = "test_weight_only_" + test_file + ".pkl"
                label_name = "test_label_only_" + test_file + ".npy"
                if not total:
                    pickle_file = "CountVectorizer_" + test_file + "_only" + ".pkl"
                else:
                    pickle_file = "Vectorizer_total_ngram_1_2.pkl"
                    file_name = "test_weight_" + test_file + ".pkl"
                    label_name = "test_label_" + test_file + ".npy"
            else:
                file_name = "test_weight_" + test_file + ".pkl"
                label_name = "test_label_" + test_file + ".npy"
                if not total:
                    pickle_file = "CountVectorizer_" + test_file + ".pkl"
                else:
                    pickle_file = "Vectorizer_total_ngram_1_2.pkl"

            if not os.path.exists(os.path.join(TEST_PATH, self.rule)):
                os.mkdir(os.path.join(TEST_PATH, self.rule))
            # 测试集特征文件不存在，生成
            if not os.path.exists(os.path.join(TEST_PATH, self.rule, label_name)):
                self.Counter = pickle.load(open(os.path.join("../../data", pickle_file), 'rb'))
                # self.Counter = pickle.load(open(os.path.join(self.path, test_file[:-1], pickle_file), 'rb'))
                self.load_test(test_file)

                print("get label...")
                self.get_label(os.path.join(TEST_PATH, self.rule, label_name))

                print("get weight...")
                token_counter = self.Counter.transform(self.data['transData.sentenceList'].values)
                weight = token_counter.toarray()
                print(weight.shape)
                pickle.dump(token_counter, open(os.path.join(TEST_PATH, self.rule, file_name), 'wb'))


if __name__ == "__main__":
    start_time = time.time()

    rule = "诋毁同事" # ""无中生有"
    max_features = 15000
    max_df = 0.5
    min_df = 3
    window = 0
    total = True
    # 训练集
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample" + str(i+1), only=False, total=total, train=True)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample" + str(i+1), only=True, total=total, train=True)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample_proportion" + str(i+1), only=False, total=total, train=True)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample_proportion" + str(i+1), only=True, total=total, train=True)

    # 测试集
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample" + str(i+1), only=True, total=total, train=False)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample" + str(i+1), only=False, total=total, train=False)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample_proportion" + str(i+1), only=True, total=total, train=False)
    for i in range(5):
        print(i+1)
        f = Features(rule, alone=False, max_df=max_df, min_df=min_df, max_features=max_features, window=window)
        f.get_weight("sample_proportion" + str(i+1), only=False, total=total, train=False)

    print('time cost is', time.time() - start_time)
