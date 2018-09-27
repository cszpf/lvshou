# encoding=utf-8
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random
import time
import os
import numpy as np
import lightgbm as lgb
import pickle
import pandas as pd
from sklearn.model_selection import KFold
sys.path.append("../project")
from interface import SupperModel
from divide import load_data, PATH1, PATH2

random.seed(2018)

PATH = "../../data/Content"
SAMPLE_PATH = "../../data/Sample"


def load_all_data(alone=False):
    rules = os.listdir(PATH)
    rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
    if not alone:
        suffix = "_agent_tokens.csv"
    else:
        suffix = "_tokens.csv"
    all_data = pd.DataFrame()
    for rule in rules:
        _ = load_data(os.path.join(PATH, rule, rule + suffix))
        all_data = pd.concat([all_data, _], axis=0)

    # 测试集样本空间
    all_data.drop_duplicates(['UUID'], inplace=True)
    all_data.reset_index(inplace=True)
    return all_data


def sample_train_data(train_label, n=5):
    """
    使用训练集中所有正样本，随机选择等数量负样本，生成n份不同的训练集，返回每份对应的下标
    :param train_label: 训练集label [[UUID, label], [UUID, label] ... [UUID, label]]
    :param n: 训练集份数
    :return: 每份训练集对应下标矩阵
    """
    result = []
    pos_idx = []
    for i, label in enumerate(train_label):
        # label[0]: UUID, label[1]: label
        if label[1] == 1:
            pos_idx.append(i)

    neg_idx = [i for i in range(len(train_label)) if i not in pos_idx]
    print("pos size", len(pos_idx))
    print("neg size", len(neg_idx))
    print()
    for i in range(n):
        _ = pos_idx.copy()
        sample_neg_idx = random.sample(neg_idx, len(pos_idx))
        _.extend(sample_neg_idx)
        result.append(_)
        print(len(result[i]))
    return np.array(result)


class Feature(object):
    def __init__(self, rule, max_df, min_df, max_features, use_idf=False):
        self.rule = rule
        self.path = os.path.join(PATH, rule)
        self.train_data = None
        self.test_data = None
        self.test_file = None
        self.seed = 2018
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.use_idf = use_idf
        self.Counter = None

    def get_label(self, data):
        label = []
        # 对每个数据样本，遍历其检测出的违规类型
        for counter in range(len(data)):
            if self.rule not in data['analysisData.illegalHitData.ruleNameList'][counter]:
                label.append(0)
            else:
                for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
                    if self.rule == item:
                        label.append(1 if data['correctInfoData.correctResult'][counter].
                                     get("correctResult")[i] == '1' else 0)
        return label

    def get_data(self, _test_file, all_data):
        self.test_file = _test_file
        test_uuid = pd.read_csv(os.path.join(SAMPLE_PATH, self.test_file + ".txt"), header=None)

        # 训练集为去掉测试集的全部数据
        train_data = all_data[~all_data['UUID'].isin(test_uuid.values[:, 0])]
        self.train_data = train_data.reset_index(drop=True)
        label = self.get_label(self.train_data)
        self.train_data['label'] = label
        self.train_data = self.train_data.sample(frac=1, random_state=self.seed)

        # 测试集由test_file指定
        test_data = all_data[all_data['UUID'].isin(test_uuid.values[:, 0])]
        self.test_data = test_data.reset_index(drop=True)
        label = self.get_label(self.test_data)
        self.test_data['label'] = label
        self.test_data = self.test_data.sample(frac=1, random_state=self.seed)

        print(len(self.train_data))
        print(len(self.test_data))
        print()

    def get_counter(self):
        if self.use_idf:
            self.Counter = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df,
                                           max_features=self.max_features, use_idf=True)
        else:
            if os.path.exists(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl")):
                print("load counter_vectorizer...")
                self.Counter = pickle.load(open(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl"), 'rb'))
                # print(sorted(self.Counter.vocabulary_.items(), key=lambda x: x[1], reverse=True))
                print(len(self.Counter.vocabulary_.items()))
            else:
                self.Counter = CountVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features)
                pickle.dump(self.Counter, open(os.path.join(PATH, self.rule, "CountVectorizer_total.pkl"), 'wb'))

    def get_feature(self, train=True):
        if train:
            token_counter = self.Counter.transform(self.train_data['transData.sentenceList'].values)
            path = os.path.join(PATH, self.rule, "stacking", "train", self.test_file)
            feature_name = "train_feature.pkl"
            label_name = "train_label.csv"
        else:
            token_counter = self.Counter.transform(self.test_data['transData.sentenceList'].values)
            path = os.path.join(PATH, self.rule, "stacking", "test", self.test_file)
            feature_name = "test_feature.pkl"
            label_name = "test_label.csv"

        weight = token_counter.toarray()
        print("weight shape", weight.shape)
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(token_counter, open(os.path.join(path, feature_name), 'wb'))
        if train:
            self.train_data[['UUID', 'label']].to_csv(os.path.join(path, label_name), sep=',',
                                                      encoding="utf-8", index=False)
        else:
            self.test_data[['UUID', 'label']].to_csv(os.path.join(path, label_name), sep=',',
                                                     encoding="utf-8", index=False)


class LGBM(SupperModel):
    def __init__(self, model, param, test_file, rule, **kags):
        super(SupperModel, self).__init__()
        self.params = param
        self.model = model(num_leaves=self.params.get("num_leaves"),
                           max_depth=self.params.get("max_depth"),
                           n_estimators=self.params.get("n_estimators"),
                           n_jobs=self.params.get("n_jobs"),
                           learning_rate=self.params.get("learning_rate"),
                           colsample_bytree=self.params.get("colsample_bytree"),
                           subsample=self.params.get("subsample"))
        self.clf = None
        self.best_iters = []
        self.test_file = test_file
        self.rule = rule

    def cv(self, X_train, y_train, k_fold, path):
        self.best_iters = []
        print("using " + str(k_fold) + " cross validation...")
        kf = KFold(n_splits=k_fold)
        preds = []
        probs = []
        counter = 0
        for train_idx, test_idx in kf.split(X_train):
            # print(train_idx, test_idx)
            counter += 1
            print("第" + str(counter) + "折交叉验证")
            X = X_train[train_idx]
            y = y_train[train_idx]
            X_val = X_train[test_idx]
            y_val = y_train[test_idx]
            print("训练集正样本数:", sum(y))
            print("测试集正样本数:", sum(y_val))

            lgb_model = self.model.fit(
                X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=self.params.get("early_stopping_rounds"),
                verbose=0)

            print("predicting...")
            test_preds = lgb_model.predict(X_val)
            test_probs = lgb_model.predict_proba(X_val)[:, 1]
            preds.extend(test_preds)
            probs.extend(test_probs)
            self.best_iters.append(lgb_model.best_iteration_)

        print("validation result...")
        self.acc(y_train, preds)
        print("best iters:")
        print(self.best_iters)

        labels = pd.DataFrame(y_train)
        preds = pd.DataFrame(preds)
        probs = pd.DataFrame(probs)

        result = pd.concat([labels, preds, probs], axis=1)
        result.to_csv(path, sep=',', encoding="utf-8", index=False)

        _path = os.path.join(PATH, self.rule, "stacking", "train", self.test_file)
        with open(os.path.join(_path, "log.txt"), 'a', encoding='utf-8') as f:
            f.write(str(self.best_iters))
            f.write('\n')
            f.write((str(sum(self.best_iters) // len(self.best_iters))))
            f.write('\n')
            f.write('\n')

    def train(self, X_train, y_train):
        self.model.n_estimators = (sum(self.best_iters) // len(self.best_iters))
        print("training...")
        print("iters:", str(self.model.n_estimators))
        self.clf = self.model.fit(X_train, y_train, verbose=1)

    def predict(self, X_test, proba=False):
        probs = self.clf.predict_proba(X_test)[:, 1]
        preds = self.clf.predict(X_test)

        if proba:
            return preds, probs
        else:
            return preds

    def acc(self, Y, Y_pred):
        """ Showing some metrics about the training process

        Parameters
        ----------
        Y : list, numpy 1-D array, pandas.Series
            The ground truth on the val dataset.
        Y_pred : list, numpy 1-D array, pandas.Series
            The predict by your model on the val dataset.
        """
        Y = list(Y); Y_pred = list(Y_pred)
        print('precision:', precision_score(Y, Y_pred))
        print('accuracy:', accuracy_score(Y, Y_pred))
        print('recall:', recall_score(Y, Y_pred))
        print('micro_F1:', f1_score(Y, Y_pred, average='micro'))
        print('macro_F1:', f1_score(Y, Y_pred, average='macro'))

        _path = os.path.join(PATH, self.rule, "stacking", "train", self.test_file)
        with open(os.path.join(_path, "log.txt"), 'a', encoding='utf-8') as f:
            f.write(str(accuracy_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(precision_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(recall_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(f1_score(Y, Y_pred, average='micro')))
            f.write('\n')
            f.write(str(f1_score(Y, Y_pred, average='macro')))
            f.write('\n')
            f.write('\n')

    def saveModel(self, save_path):
        """ Save the model during training process

        Parameters
        ----------
        save_path : str
            the model's save_path
        """
        if not os.path.exists('/'.join(os.path.split(save_path)[:-1])):
            os.makedirs('/'.join(os.path.split(save_path)[:-1]))
        with open(save_path, 'wb') as fw:
            pickle.dump(self.clf, fw)


class Stacking(SupperModel):
    def __init__(self, model, param, test_file, rule, **kags):
        super(SupperModel, self).__init__()
        self.params = param
        self.model = model(num_leaves=self.params.get("num_leaves"),
                           max_depth=self.params.get("max_depth"),
                           n_estimators=self.params.get("n_estimators"),
                           n_jobs=self.params.get("n_jobs"),
                           learning_rate=self.params.get("learning_rate"),
                           colsample_bytree=self.params.get("colsample_bytree"),
                           subsample=self.params.get("subsample"))
        self.clf = None
        self.best_iters = []
        self.test_file = test_file
        self.rule = rule

    def cv(self, X_train, y_train, k_fold, path):
        self.best_iters = []
        print("using " + str(k_fold) + " cross validation...")
        kf = KFold(n_splits=k_fold)
        preds = []
        probs = []
        counter = 0
        for train_idx, test_idx in kf.split(X_train):
            # print(train_idx, test_idx)
            counter += 1
            print("第" + str(counter) + "折交叉验证")
            X = X_train[train_idx]
            y = y_train[train_idx]
            X_val = X_train[test_idx]
            y_val = y_train[test_idx]

            lgb_model = self.model.fit(
                X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=self.params.get("early_stopping_rounds"),
                verbose=0)

            print("predicting...")
            test_preds = lgb_model.predict(X_val)
            test_probs = lgb_model.predict_proba(X_val)[:, 1]
            preds.extend(test_preds)
            probs.extend(test_probs)
            self.best_iters.append(lgb_model.best_iteration_)

        print("validation result...")
        self.acc(y_train, preds)
        print("best iters:")
        print(self.best_iters)

        labels = pd.DataFrame(y_train)
        preds = pd.DataFrame(preds)
        probs = pd.DataFrame(probs)

        result = pd.concat([labels, preds, probs], axis=1)
        result.to_csv(path, sep=',', encoding="utf-8", index=False)

        _path = os.path.join(PATH, self.rule, "stacking", "test", self.test_file)
        with open(os.path.join(_path, "layer2_log.txt"), 'a', encoding='utf-8') as f:
            f.write(str(self.best_iters))
            f.write('\n')
            f.write((str(sum(self.best_iters) // len(self.best_iters))))
            f.write('\n')
            f.write('\n')

    def train(self, X_train, y_train):
        self.model.n_estimators = (sum(self.best_iters) // len(self.best_iters))
        print("training...")
        print("iters:", str(self.model.n_estimators))
        self.clf = self.model.fit(X_train, y_train, verbose=0)

    def predict(self, X_test, proba=False):
        probs = self.clf.predict_proba(X_test)[:, 1]
        preds = self.clf.predict(X_test)

        if proba:
            return preds, probs
        else:
            return preds

    def acc(self, Y, Y_pred):
        """ Showing some metrics about the training process

        Parameters
        ----------
        Y : list, numpy 1-D array, pandas.Series
            The ground truth on the val dataset.
        Y_pred : list, numpy 1-D array, pandas.Series
            The predict by your model on the val dataset.
        """
        Y = list(Y); Y_pred = list(Y_pred)
        print('precision:', precision_score(Y, Y_pred))
        print('accuracy:', accuracy_score(Y, Y_pred))
        print('recall:', recall_score(Y, Y_pred))
        print('micro_F1:', f1_score(Y, Y_pred, average='micro'))
        print('macro_F1:', f1_score(Y, Y_pred, average='macro'))

        _path = os.path.join(PATH, self.rule, "stacking", "test", self.test_file)
        with open(os.path.join(_path, "layer2_log.txt"), 'a', encoding='utf-8') as f:
            f.write(str(accuracy_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(precision_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(recall_score(Y, Y_pred)))
            f.write('\n')
            f.write(str(f1_score(Y, Y_pred, average='micro')))
            f.write('\n')
            f.write(str(f1_score(Y, Y_pred, average='macro')))
            f.write('\n')
            f.write('\n')


def first_layer(rule, param):
    # 加载所有数据
    data = load_all_data(alone=False)

    for i in range(5):
        test_file = "sample" + str(i + 1)
        n = 5
        model = LGBM(lgb.LGBMClassifier, param, rule=rule, test_file=test_file)
        path = os.path.join(PATH, rule, "stacking", "train", test_file)
        test_path = os.path.join(PATH, rule, "stacking", "test", test_file)

        # 生成训练集，测试集特征及label，保存
        if not os.path.exists(os.path.join(path, "train_label.csv")):
            print("loading data...")
            print(data.shape)
            f = Feature(rule, max_df=0.5, min_df=3, max_features=10000)
            f.get_data(test_file, data)
            f.get_counter()
            f.get_feature(train=True)
            f.get_feature(train=False)

        # 加载训练集特征，label
        train_feature = pickle.load(open(os.path.join(path, "train_feature.pkl"), 'rb'))
        train_label = pd.read_csv(os.path.join(path, "train_label.csv"),
                                  sep=',', encoding="utf-8")
        # train_feature = train_feature.toarray()
        train_label = train_label.values

        # 加载测试集，label
        test_feature = pickle.load(open(os.path.join(test_path, "test_feature.pkl"), 'rb'))
        test_label = pd.read_csv(os.path.join(test_path, "test_label.csv"), sep=',', encoding="utf-8")
        test_label = test_label.values

        print("all train data size", train_feature.shape)
        print("all train label size", train_label.shape)
        print()

        # 训练集，验证集划分
        if os.path.exists(os.path.join(path, "train_idx.npy")):
            train_idx = np.load(os.path.join(path, "train_idx.npy"))
            eval_idx = np.load(os.path.join(path, "eval_idx.npy"))
        else:
            train_idx = random.sample(range(len(train_label)), k=int(len(train_label) * 0.8))
            eval_idx = [i for i in range(len(train_label)) if i not in train_idx]
            np.save(os.path.join(path, "train_idx.npy"), train_idx)
            np.save(os.path.join(path, "eval_idx.npy"), eval_idx)

        train_data, eval_data = train_feature[train_idx], train_feature[eval_idx]
        train_label, eval_label = train_label[train_idx], train_label[eval_idx]

        print("train size", train_data.shape, "train label", train_label.shape)
        print("eval size", eval_data.shape, "eval label", eval_label.shape)
        print()

        # 记录训练集，验证集大小
        with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
            f.write(test_file)
            f.write('\n')
            f.write("train size: " + str(train_data.shape) + " train label: " + str(train_label.shape))
            f.write('\n')
            f.write("eval size: " + str(eval_data.shape) + " eval label: " + str(eval_label.shape))
            f.write('\n')
            f.write("test size: " + str(test_feature.shape) + " eval label: " + str(test_label.shape))
            f.write('\n')
            f.write('\n')

        # 使用训练集中所有正样本，随机选择5份等数量的负样本，构成5份训练集
        if os.path.exists(os.path.join(path, "train_data_" + str(n) + ".npy")):
            sample_result = np.load(os.path.join(path, "train_data_" + str(n) + ".npy"))
        else:
            sample_result = sample_train_data(train_label, n)
            np.save(os.path.join(path, "train_data_" + str(n) + ".npy"), sample_result)
        print("sample size", sample_result.shape)

        # 针对5份训练集，训练5个模型
        for model_counter in range(n):

            # 获得第i份训练集
            random.shuffle(sample_result[model_counter])
            _train_data = train_data[sample_result[model_counter]].toarray()
            _train_label = train_label[sample_result[model_counter]][:, 1].astype(int)

            # 交叉验证获得最优训练次数，保存交叉验证结果
            model.cv(_train_data, _train_label, k_fold=5, path=os.path.join(path, "train_" + str(model_counter + 1)
                                                                            + "_cv.csv"))

            # 使用训练集数据训练
            model.train(_train_data, _train_label)

            # 预测验证集结果并保存
            with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
                f.write("eval result")
                f.write('\n')
            eval_preds, eval_probs = model.predict(eval_data.toarray(), proba=True)
            model.saveModel(os.path.join(path, "model_" + str(model_counter + 1) + ".h5"))
            model.acc(eval_label[:, 1].astype(int), eval_preds)

            eval_result = pd.concat([pd.DataFrame(eval_label), pd.DataFrame(eval_probs)], axis=1)

            eval_result.to_csv(os.path.join(path, "train_" + str(model_counter + 1) + "_result.csv"), sep=',',
                               encoding="utf-8", index=False)

            # 预测测试集结果并保存
            with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
                f.write("test result")
                f.write('\n')
            test_preds, test_probs = model.predict(test_feature.toarray(), proba=True)
            model.acc(test_label[:, 1].astype(int), test_preds)

            test_result = pd.concat([pd.DataFrame(test_label), pd.DataFrame(test_probs)], axis=1)

            test_result.to_csv(os.path.join(test_path, "test_" + str(model_counter + 1) + "_result.csv"), sep=',',
                               encoding="utf-8", index=False)

    for i in range(5):
        test_file = "sample_proportion" + str(i + 1)
        n = 5
        model = LGBM(lgb.LGBMClassifier, param, rule=rule, test_file=test_file)
        path = os.path.join(PATH, rule, "stacking", "train", test_file)
        test_path = os.path.join(PATH, rule, "stacking", "test", test_file)

        # 生成训练集，测试集特征及label，保存
        if not os.path.exists(os.path.join(path, "train_label.csv")):
            print("loading data...")
            print(data.shape)
            f = Feature(rule, max_df=0.5, min_df=3, max_features=10000)
            f.get_data(test_file, data)
            f.get_counter()
            f.get_feature(train=True)
            f.get_feature(train=False)

        # 加载训练集特征，label
        train_feature = pickle.load(open(os.path.join(path, "train_feature.pkl"), 'rb'))
        train_label = pd.read_csv(os.path.join(path, "train_label.csv"),
                                  sep=',', encoding="utf-8")
        # train_feature = train_feature.toarray()
        train_label = train_label.values

        # 加载测试集，label
        test_feature = pickle.load(open(os.path.join(test_path, "test_feature.pkl"), 'rb'))
        test_label = pd.read_csv(os.path.join(test_path, "test_label.csv"), sep=',', encoding="utf-8")
        test_label = test_label.values

        print("all train data size", train_feature.shape)
        print("all train label size", train_label.shape)
        print()

        # 训练集，验证集划分
        if os.path.exists(os.path.join(path, "train_idx.npy")):
            train_idx = np.load(os.path.join(path, "train_idx.npy"))
            eval_idx = np.load(os.path.join(path, "eval_idx.npy"))
        else:
            train_idx = random.sample(range(len(train_label)), k=int(len(train_label) * 0.8))
            eval_idx = [i for i in range(len(train_label)) if i not in train_idx]
            np.save(os.path.join(path, "train_idx.npy"), train_idx)
            np.save(os.path.join(path, "eval_idx.npy"), eval_idx)

        train_data, eval_data = train_feature[train_idx], train_feature[eval_idx]
        train_label, eval_label = train_label[train_idx], train_label[eval_idx]

        print("train size", train_data.shape, "train label", train_label.shape)
        print("eval size", eval_data.shape, "eval label", eval_label.shape)
        print()

        # 记录训练集，验证集大小
        with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
            f.write(test_file)
            f.write('\n')
            f.write("train size: " + str(train_data.shape) + " train label: " + str(train_label.shape))
            f.write('\n')
            f.write("eval size: " + str(eval_data.shape) + " eval label: " + str(eval_label.shape))
            f.write('\n')
            f.write('\n')

        # 使用训练集中所有正样本，随机选择5份等数量的负样本，构成5份训练集
        if os.path.exists(os.path.join(path, "train_data_" + str(n) + ".npy")):
            sample_result = np.load(os.path.join(path, "train_data_" + str(n) + ".npy"))
        else:
            sample_result = sample_train_data(train_label, n)
            np.save(os.path.join(path, "train_data_" + str(n) + ".npy"), sample_result)
        print("sample size", sample_result.shape)

        # 针对5份训练集，训练5个模型
        for model_counter in range(n):
            # 获得第i份训练集
            random.shuffle(sample_result[model_counter])
            _train_data = train_data[sample_result[model_counter]].toarray()
            _train_label = train_label[sample_result[model_counter]][:, 1].astype(int)

            # 交叉验证获得最优训练次数，保存交叉验证结果
            model.cv(_train_data, _train_label, k_fold=5, path=os.path.join(path, "train_" + str(model_counter + 1)
                                                                            + "_cv.csv"))

            # 使用训练集数据训练
            model.train(_train_data, _train_label)

            # 预测验证集结果并保存
            with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
                f.write("eval result")
                f.write('\n')
            eval_preds, eval_probs = model.predict(eval_data.toarray(), proba=True)
            model.saveModel(os.path.join(path, "model_" + str(model_counter + 1) + ".h5"))
            model.acc(eval_label[:, 1].astype(int), eval_preds)

            eval_result = pd.concat([pd.DataFrame(eval_label), pd.DataFrame(eval_probs)], axis=1)

            eval_result.to_csv(os.path.join(path, "train_" + str(model_counter + 1) + "_result.csv"), sep=',',
                               encoding="utf-8", index=False)

            # 预测测试集结果并保存
            with open(os.path.join(path, "log.txt"), 'a', encoding='utf-8') as f:
                f.write("test result")
                f.write('\n')
            test_preds, test_probs = model.predict(test_feature.toarray(), proba=True)
            model.acc(test_label[:, 1].astype(int), test_preds)

            test_result = pd.concat([pd.DataFrame(test_label), pd.DataFrame(test_probs)], axis=1)

            test_result.to_csv(os.path.join(test_path, "test_" + str(model_counter + 1) + "_result.csv"), sep=',',
                               encoding="utf-8", index=False)


def second_layer(rule, param):
    for i in range(5):
        test_file = "sample" + str(i+1)
        # test_file = "sample_proportion" + str(i + 1)
        model = Stacking(lgb.LGBMClassifier, param, rule=rule, test_file=test_file)
        train_path = os.path.join(PATH, rule, "stacking", "train", test_file)
        test_path = os.path.join(PATH, rule, "stacking", "test", test_file)

        # 加载第一层训练集结果
        train_preds = pd.DataFrame()
        for model_counter in range(5):
            _ = pd.read_csv(os.path.join(train_path, "train_" + str(model_counter + 1) + "_result.csv"), sep=',',
                            encoding='utf-8')
            _.columns = ["UUID", "label", "pred_" + str(model_counter + 1)]
            if 'UUID' not in train_preds:
                train_preds['UUID'] = _['UUID']
            if 'label' not in train_preds:
                train_preds['label'] = _['label']
            train_preds["pred_" + str(model_counter + 1)] = _["pred_" + str(model_counter + 1)]
        train_preds.to_csv(os.path.join(test_path, "train_pred.csv"), sep=',', index=False, encoding='utf-8')

        # 加载第一层测试集结果
        test_preds = pd.DataFrame()
        for model_counter in range(5):
            _ = pd.read_csv(os.path.join(test_path, "test_" + str(model_counter + 1) + "_result.csv"), sep=',',
                            encoding='utf-8')
            _.columns = ["UUID", "label", "pred_" + str(model_counter + 1)]
            if 'UUID' not in test_preds:
                test_preds['UUID'] = _['UUID']
            if 'label' not in test_preds:
                test_preds['label'] = _['label']
            test_preds["pred_" + str(model_counter + 1)] = _["pred_" + str(model_counter + 1)]
        test_preds.to_csv(os.path.join(test_path, "test_pred.csv"), sep=',', index=False, encoding='utf-8')

        model.cv(train_preds[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values,
                 train_preds['label'].values, k_fold=5, path=os.path.join(test_path, "stacking_cv.csv"))
        model.train(train_preds[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values,
                    train_preds['label'].values)
        model.saveModel(os.path.join(test_path, "model_stacking.h5"))
        preds = model.predict(test_preds[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values)
        model.acc(test_preds['label'].values, preds)


if __name__ == "__main__":
    start_time = time.time()

    param = {
        "num_leaves": 35,
        "max_depth": 7,
        "n_estimators": 20000,
        "n_jobs": 20,
        "learning_rate": 0.01,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 100,
    }

    rule = "敏感词"
    # first_layer(rule, param)
    second_layer(rule, param)
    print('time cost is', time.time() - start_time)


