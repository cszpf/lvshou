# encoding=utf-8
import sys
import time
import os
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
sys.path.append("..")
from project.interface import SupperModel
from sklearn.model_selection import KFold
from project.divide import load_data as ld

PATH = "../../data/Content"
SAMPLE_PATH = "../../data/Sample"


def load_data(rule, test_file, total=False, only=True, train=True):
    path = os.path.join(PATH, rule, test_file[:-1])
    if train:
        if only:
            token_counter = pickle.load(
                open(os.path.join(path, rule + "_train_weight_only_" + test_file + ".pkl"), 'rb'))
            weight = token_counter.toarray()
            label = np.load(os.path.join(path, rule + "_train_label_only_" + test_file + ".npy"))
        else:
            token_counter = pickle.load(
                open(os.path.join(path, rule + "_train_weight_" + test_file + ".pkl"), 'rb'))
            weight = token_counter.toarray()
            label = np.load(os.path.join(path, rule + "_train_label_" + test_file + ".npy"))
    else:
        if total:
            token_counter = pickle.load(open(os.path.join(SAMPLE_PATH, rule, "test_weight_" + test_file + ".pkl"), 'rb'))
            weight = token_counter.toarray()
            label = np.load(os.path.join(SAMPLE_PATH, rule, "test_label_" + test_file + ".npy"))
        else:
            if only:
                token_counter = pickle.load(
                    open(os.path.join(SAMPLE_PATH, rule, "test_weight_only_" + test_file + ".pkl"), 'rb'))
                weight = token_counter.toarray()
                label = np.load(os.path.join(SAMPLE_PATH, rule, "test_label_only_" + test_file + ".npy"))
            else:
                token_counter = pickle.load(
                    open(os.path.join(SAMPLE_PATH, rule, "test_weight_" + test_file + ".pkl"), 'rb'))
                weight = token_counter.toarray()
                label = np.load(os.path.join(SAMPLE_PATH, rule, "test_label_" + test_file + ".npy"))
    return weight, label


class LGBM(SupperModel):
    def __init__(self, model, param, **kags):
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

    def cv(self, X_train, y_train, k_fold, test_file, only, rule):

        if only:
            if test_file[:-1] == "sample":
                path = os.path.join(PATH, rule, "result", "sample_true")
            else:
                path = os.path.join(PATH, rule, "result", "sample_proportion_true")
        else:
            if test_file[:-1] == "sample":
                path = os.path.join(PATH, rule, "result", "sample_false")
            else:
                path = os.path.join(PATH, rule, "result", "sample_proportion_false")

        if not os.path.exists(path):
            os.makedirs(path)

        print("using " + str(k_fold) + " cross validation...")
        kf = KFold(n_splits=k_fold)
        preds = []
        probs = []
        for train_idx, test_idx in kf.split(X_train):
            print(train_idx, test_idx)
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

        pickle.dump(preds, open(os.path.join(path, test_file + "_preds_Vectorizer_total_ngram_1_2.pkl"), 'wb'))
        pickle.dump(probs, open(os.path.join(path, test_file + "_probs_Vectorizer_total_ngram_1_2.pkl"), 'wb'))

        print("validation result...")
        self.acc(y_train, preds)
        print("best iters:")
        print(self.best_iters)
        with open(os.path.join(PATH, rule, "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
            f.write((str(sum(self.best_iters) // len(self.best_iters))))
            f.write('\n')
            f.write('\n')

    def train(self, X_train, y_train):
        self.model.n_estimators = (sum(self.best_iters) // len(self.best_iters))
        # self.model.n_estimators = n_estimators
        print("training...")
        print("iters:", str(self.model.n_estimators))
        self.clf = self.model.fit(X_train, y_train, verbose=1)

    def predict(self, X_test, proba=True):
        if proba:
            preds = []
            probs = self.clf.predict_proba(X_test)[:, 1]
            for i in probs:
                if i > 0.5:
                    preds.append(1)
                else:
                    preds.append(0)
            return preds, probs
        else:
            return self.clf.predict(X_test)

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

        with open(os.path.join(PATH, "夸大肥胖后果", "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
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


def load_test(test_file, alone):
    import pandas as pd
    test_uuid = pd.read_csv(os.path.join('../../data/Sample', test_file + ".txt"), header=None)
    rules = os.listdir(PATH)
    rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
    if not alone:
        suffix = "_agent_tokens.csv"
    else:
        suffix = "_tokens.csv"
    test_data = pd.DataFrame()
    for rule in rules:
        _ = ld(os.path.join(PATH, rule, rule + suffix))
        test_data = pd.concat([test_data, _], axis=0)

    # 测试集样本空间
    test_data.drop_duplicates(['UUID'], inplace=True)
    test_data.reset_index(inplace=True)
    print(len(test_data))
    data = test_data[test_data['UUID'].isin(test_uuid.values[:, 0])]
    data.reset_index(drop=True, inplace=True)
    return data


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
    # test_file = "sample_proportion"

    rule = "夸大肥胖后果"
    total = True
    for i in range(3):
        test_file = "sample" + str(i+1)
        X_train, y_train = load_data(rule, test_file=test_file, total=total, only=False, train=True)
        X_test, y_test = load_data(rule, test_file=test_file, total=total, only=False, train=False)
    
        print("Train:", X_train.shape)
        print("Test:", X_test.shape)
        with open(os.path.join(PATH, rule, "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
            f.write(str(X_train.shape))
            f.write('\n')
            f.write(str(X_test.shape))
            f.write('\n')
            f.write('\n')
        model = LGBM(lgb.LGBMClassifier, param)
        model.cv(X_train, y_train, 3, test_file, only=False, rule=rule)
        model.train(X_train, y_train)
        preds, probs = model.predict(X_test, proba=True)
        model.acc(y_test, preds)
    
        test_data = load_test(test_file, alone=False)
        test_data['result'] = y_test
        test_data['prob'] = probs
        test_data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'result', 'prob']]\
            .to_csv(os.path.join(PATH, rule, test_file + "_pred.csv"), sep=',', encoding='utf-8')

    for i in range(3):
        test_file = "sample" + str(i+1)
        X_train, y_train = load_data(rule, test_file=test_file, total=total, only=True, train=True)
        X_test, y_test = load_data(rule, test_file=test_file, total=total, only=True, train=False)

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)
        with open(os.path.join(PATH, rule, "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
            f.write(str(X_train.shape))
            f.write('\n')
            f.write(str(X_test.shape))
            f.write('\n')
            f.write('\n')
        model = LGBM(lgb.LGBMClassifier, param)
        model.cv(X_train, y_train, 3, test_file, only=True, rule=rule)
        model.train(X_train, y_train)
        preds, probs = model.predict(X_test, proba=True)
        model.acc(y_test, preds)

        test_data = load_test(test_file, alone=False)
        test_data['result'] = y_test
        test_data['prob'] = probs
        test_data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'result', 'prob']]\
            .to_csv(os.path.join(PATH, rule, test_file + "_pred_only.csv"), sep=',', encoding='utf-8')

    for i in range(3):
        test_file = "sample_proportion" + str(i+1)
        X_train, y_train = load_data(rule, test_file=test_file, total=total, only=False, train=True)
        X_test, y_test = load_data(rule, test_file=test_file, total=total, only=False, train=False)

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)
        with open(os.path.join(PATH, rule, "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
            f.write(str(X_train.shape))
            f.write('\n')
            f.write(str(X_test.shape))
            f.write('\n')
            f.write('\n')
        model = LGBM(lgb.LGBMClassifier, param)
        model.cv(X_train, y_train, 3, test_file, only=False, rule=rule)
        model.train(X_train, y_train)
        preds, probs = model.predict(X_test, proba=True)
        model.acc(y_test, preds)

        test_data = load_test(test_file, alone=False)
        test_data['result'] = y_test
        test_data['prob'] = probs
        test_data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'result', 'prob']]\
            .to_csv(os.path.join(PATH, rule, test_file + "_pred.csv"), sep=',', encoding='utf-8')

    for i in range(3):
        test_file = "sample_proportion" + str(i+1)
        X_train, y_train = load_data(rule, test_file=test_file, total=total, only=True, train=True)
        X_test, y_test = load_data(rule, test_file=test_file, total=total, only=True, train=False)

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)
        with open(os.path.join(PATH, rule, "window_sample_f_t_pro_f_t.txt"), 'a', encoding='utf-8') as f:
            f.write(str(X_train.shape))
            f.write('\n')
            f.write(str(X_test.shape))
            f.write('\n')
            f.write('\n')
        model = LGBM(lgb.LGBMClassifier, param)
        model.cv(X_train, y_train, 3, test_file, only=True, rule=rule)
        model.train(X_train, y_train)
        preds, probs = model.predict(X_test, proba=True)
        model.acc(y_test, preds)

        test_data = load_test(test_file, alone=False)
        test_data['result'] = y_test
        test_data['prob'] = probs
        test_data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'result', 'prob']]\
            .to_csv(os.path.join(PATH, rule, test_file + "_pred_only.csv"), sep=',', encoding='utf-8')

    print('time cost is', time.time() - start_time)
