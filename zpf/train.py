import numpy as np
np.random.seed(2018)
import pickle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.ensemble import BaggingClassifier
import lightgbm as lgb
from Stacking import SBBTree
import os

PATH_TN = '../../../zhijian_data/Token'
CV = 2
def acc(clf, Y, Y_pred):
    Y = list(Y); Y_pred = list(Y_pred)
    print(clf + 'precision:', precision_score(Y, Y_pred))
    # print(clf + 'accuracy:', accuracy_score(Y, Y_pred))
    print(clf + 'recall:', recall_score(Y, Y_pred))
    print(clf + 'micro_F1:', f1_score(Y, Y_pred, average='micro'))
    print(clf + 'macro_F1:', f1_score(Y, Y_pred, average='macro'))

class Model(object):
    def __init__(self, path_train=PATH_TN,  learning_rate=0.02, n_estimators=200,
                 max_depth=10, min_child_weight=1, gamma=0.1, subsample=0.5, colsample_bytree=0.6,
                 objective='binary:logistic', seed=2018, label=''):
        self.xgboost = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
	        min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
	        objective=objective, seed=seed, n_jobs=-1)
        # self.xgboost = BaggingClassifier(self.xgboost, random_state=2018, n_jobs=-1)
        self.gbdt = GradientBoostingClassifier()
        self.rf = RandomForestClassifier()
        self.label = label
        self.gbms = []


    def writeFile(self, tests, preds, uuids):
        if not os.path.exists('./setting/model'):
            os.makedirs('./setting/model')
        with open('./setting/model/{}_BDC.txt'.format(self.label), 'w+') as f:
            f.write('uuid,test,pre\n')
            for i, j, k in zip(uuids, tests, preds):
                if j != k:
                    f.write("{},{},{}\n".format(i, j, k))

    def generalCV(self, index, train_data, train_label, train_uuid):
        train_data = train_data[index]
        train_label = np.array(train_label)[index]; train_uuid = np.array(train_uuid)[index]
        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=2018)
        preds, tests, uuids = [], [], []
        print(train_label)
        for k, (train_index, test_index) in enumerate(skf.split(train_label, train_label)):
            X_train = train_data[train_index]; y_train = train_label[train_index]
            X_test = train_data[test_index]; y_test = train_label[test_index]
            self.gbm = self.xgboost.fit(X_train, y_train)
            pred_y = self.gbm.predict(X_test)
            # print('第{}折的实验结果:'.format(k))
            # acc('%%', y_test, pred_y.round())
            self.gbms.append(self.gbm)
            preds.extend(pred_y.round()); tests.extend(y_test); uuids.extend(train_uuid[test_index])
        print('汇总之后的实验结果为:')
        acc('%%', tests, preds)
        self.writeFile(tests, preds, uuids)

    def predict(self, test_data, test_label, test_uuid):
        test_pred = 0
        for i in self.gbms:
            test_pred += np.array(i.predict(test_data))
        print('测试集的结果为：')
        acc('%%', test_label, (test_pred/len(self.gbms)).round())
        self.writeFile(np.array(test_label), test_pred.round(), np.array(test_uuid))


    # def stacking(self, index):
    #     train_index, test_index = train_test_split(index, test_size=0.4, random_state=2017)
    #     X_train, X_test = self._X[train_index], self._X[test_index]
    #     Y_train, Y_test = self._Y[train_index], self._Y[test_index]
    #     uuid_test = self.uuid[test_index]
    #     # print(np.shape(X_train), np.shape(Y_train))
    #     print('avg(auc):', self.SBBTree.fit(X_train, Y_train))
    #     Y_pred = self.SBBTree.predict(X_test)
    #     acc('dev', Y_test, Y_pred.round())
    #     self.writeFile(Y_test, Y_pred.round(), uuid_test)


    def evl(self, train_data, train_label, train_uuid,\
            test_data, test_label, test_uuid):
        index = list(range(np.shape(train_data)[0]))
        for _ in range(9):
            np.random.shuffle(index)
        # self.stacking(index)
        self.generalCV(index, train_data, train_label, train_uuid)
        self.predict(test_data, test_label, test_uuid)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            mdl = self.xgboost
            pickle.dump(mdl, f)

    def load(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.xgboost = pickle.load(f)


if __name__ == "__main__":
    import time
    start = time.time()
    # mymodel = Model(label='无中生有', debug=True, bdc=True, qz='', opt='bdc')
    # mymodel.evl()
    # # mymodel.save('./setting/model/xgboost.pk')
    print('本次训练耗时:{}'.format(time.time()-start))