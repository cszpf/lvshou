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

random.seed(2018)

PATH = "../../data/Content"
SAMPLE_PATH = "../../data/Sample"


def load_data(rule, only, test_file, train=True):
    if train:
        path = os.path.join(PATH, rule, "stacking", "train")
        if only:
            dir_name = test_file[:-1] + "_true"
            label_name = rule + "_train_label_only_" + test_file + ".npy"
        else:
            dir_name = test_file[:-1] + "_false"
            label_name = rule + "_train_label_" + test_file + ".npy"
        label = np.load(os.path.join(path, dir_name, label_name))
        train_files = os.listdir(os.path.join(path, dir_name))
        train_data = pd.DataFrame()
        train_data['label'] = label
        for train_file in train_files[:-1]:
            data = pickle.load(open(os.path.join(path, dir_name, train_file), 'rb'))
            if test_file[:-1] == "sample":
                train_data[train_file[14:-4]] = data
            else:
                train_data[train_file[25:-4]] = data
        return train_data
    else:
        path = os.path.join(PATH, rule, "stacking", "test")
        test_dirs = os.listdir(path)
        if only:
            file_name = test_file + "_pred_only.csv"
        else:
            file_name = test_file + "_pred.csv"
        test_data = pd.DataFrame()
        for dir in test_dirs:
            data = pd.read_csv(os.path.join(path, dir, file_name), encoding='utf-8', sep=',')
            if 'label' not in test_data.columns:
                test_data['label'] = data['result']
            if 'UUID' not in test_data.columns:
                test_data['UUID'] = data['UUID']
            test_data[dir] = data['prob']
        return test_data


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
            print(train_idx, test_idx)
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

        _path = os.path.join(PATH, self.rule, "stacking", "train")
        # with open(os.path.join(_path, "layer2_log.txt"), 'a', encoding='utf-8') as f:
        #     f.write(str(self.best_iters))
        #     f.write('\n')
        #     f.write((str(sum(self.best_iters) // len(self.best_iters))))
        #     f.write('\n')
        #     f.write('\n')

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

        _path = os.path.join(PATH, self.rule, "stacking", "train")
        # with open(os.path.join(_path, "layer2_log.txt"), 'a', encoding='utf-8') as f:
        #     f.write(str(accuracy_score(Y, Y_pred)))
        #     f.write('\n')
        #     f.write(str(precision_score(Y, Y_pred)))
        #     f.write('\n')
        #     f.write(str(recall_score(Y, Y_pred)))
        #     f.write('\n')
        #     f.write(str(f1_score(Y, Y_pred, average='micro')))
        #     f.write('\n')
        #     f.write(str(f1_score(Y, Y_pred, average='macro')))
        #     f.write('\n')
        #     f.write('\n')


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

    rule = "部门名称"
    # test_file = "sample_proportion1"
    test_file = "sample1"
    only = True
    if only:
        prefix = test_file + "_true_"
    else:
        prefix = test_file + "_false_"
    train_data = load_data(rule, only=only, test_file=test_file, train=True)
    test_data = load_data(rule, only=only, test_file=test_file, train=False)
    print(train_data)
    print(test_data)
    # train_data.to_csv(os.path.join(PATH, rule, "stacking", "train", prefix + "train.csv"), sep=',',
    #                   encoding='utf-8', index=False)
    # test_data['pred'] = test_data['Vectorizer_total_ngram_1'] * 0.166 + test_data['Vectorizer_total_ngram_2'] * 0.166 + \
    #                     test_data['window_20_Vectorizer_total_ngram_1_2'] * 0.166 + \
    #                     test_data['window_20_Vectorizer_total_ngram_1_3'] * 0.166 + \
    #                     test_data['window_20_Vectorizer_total_ngram_2_3'] * 0.166 + \
    #                     test_data['window_20_Vectorizer_total_ngram_3'] * 0.166
    # preds = []
    # for i in test_data['pred']:
    #     if i > 0.5:
    #         preds.append(1)
    #     else:
    #         preds.append(0)
    # print(test_data[['label', 'pred']])
    model = Stacking(lgb.LGBMClassifier, param, rule=rule, test_file=test_file)
    # model.acc(test_data['label'], preds)
    model.cv(train_data[['Vectorizer_total_ngram_1_1',
                         'Vectorizer_total_ngram_2_2',
                         'Vectorizer_total_ngram_3_3',
                         'Vectorizer_total_ngram_1_2',
                         'Vectorizer_total_ngram_1_3',
                         'Vectorizer_total_ngram_2_3']].values,
             train_data['label'].values, k_fold=5, path=os.path.join(PATH, rule, "stacking",
                                                                     "train", prefix + "stacking_cv.csv"))
    model.train(train_data[['Vectorizer_total_ngram_1_1',
                            'Vectorizer_total_ngram_2_2',
                            'Vectorizer_total_ngram_3_3',
                            'Vectorizer_total_ngram_1_2',
                            'Vectorizer_total_ngram_1_3',
                            'Vectorizer_total_ngram_2_3']].values,
                train_data['label'].values)
    model.saveModel(os.path.join(PATH, rule, "stacking", "train", prefix + "model_stacking.h5"))
    preds = model.predict(test_data[['Vectorizer_total_ngram_1_1',
                                     'Vectorizer_total_ngram_2_2',
                                     'Vectorizer_total_ngram_3_3',
                                     'Vectorizer_total_ngram_1_2',
                                     'Vectorizer_total_ngram_1_3',
                                     'Vectorizer_total_ngram_2_3']].values)
    test_data['pred'] = preds
    test_data[['UUID', 'pred', 'label']].to_csv(os.path.join(PATH, rule, "stacking", "train", prefix + "pred.csv"), sep=',', encoding='utf-8')
    model.acc(test_data['label'].values, preds)
    print('time cost is', time.time() - start_time)
