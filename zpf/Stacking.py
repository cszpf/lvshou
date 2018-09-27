import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score,recall_score,f1_score

def acc(clf, Y, Y_pred):
    Y = list(Y); Y_pred = list(Y_pred)
    print(clf + 'accuracy:', accuracy_score(Y, Y_pred))
    print(clf + 'recall:', recall_score(Y, Y_pred))
    print(clf + 'micro_F1:', f1_score(Y, Y_pred, average='micro'))
    print(clf + 'macro_F1:', f1_score(Y, Y_pred, average='macro'))

class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    def __init__(self, model, bagging_num, num_boost_round=20000, early_stopping_rounds=50):
        self.lgb_params = {
            'num_leaves': 50,
            'max_depth': 8,
            'subsample': 0.85,
            'subsample_freq': 1,
            'verbosity': -1,
            'colsample_bytree': 0.85,
            'min_child_weight': 50,
            'nthread': 4,
            'seed': 2017,
            'boosting_type': 'rf',
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'is_unbalance': True,
            'lambda_l1': 0.5,
            'lambda_l2': 35
        }
        self.bagging_num = bagging_num
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = model
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        if len(self.model) >= 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=len(self.model), shuffle=True, random_state=1)
            for _, model in enumerate(self.model):
                for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                    X_train = X[train_index]
                    y_train = y[train_index]
                    X_test = X[test_index]
                    y_test = y[test_index]
                    if _ == 0 or _ == len(self.model)-1:
                        lgb_train = model.Dataset(X_train, y_train)
                        lgb_eval = model.Dataset(X_test, y_test, reference=lgb_train)
                        gbm = model.train(self.lgb_params,lgb_train,num_boost_round=self.num_boost_round,
                        valid_sets = lgb_eval,early_stopping_rounds=self.early_stopping_rounds,verbose_eval=False)
                        pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                    else:
                        gbm = model.fit(X_train, y_train)
                        pred_y = gbm.predict(X_test)
                    self.stacking_model.append(gbm)
                    layer_train[test_index, 1] = pred_y
                X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        cv_auc = []
        self.SK_b = StratifiedKFold(n_splits=self.bagging_num, shuffle=True, random_state=1)
        for _, model in enumerate(self.model):
            for k, (train_index, test_index) in enumerate(self.SK_b.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]
                if _ == 0 or _ == len(self.model)-1:
                    lgb_train = lgb.Dataset(X_train, y_train)
                    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
                    gbm = lgb.train(self.lgb_params,lgb_train,num_boost_round=self.num_boost_round,
                    valid_sets=lgb_eval,early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
                    pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                else:
                    gbm = model.fit(X_train, y_train)
                    pred_y = gbm.predict(X_test)
                self.bagging_model.append(gbm)
                print('第{}次第{}折的实验结果:'.format(_, k))
                acc('%%', y_test, pred_y.round())
                cv_auc.append(roc_auc_score(y_test, pred_y))
        return np.mean(cv_auc)

    def predict(self, X_pred):
        """ predict test data. """
        print(len(self.stacking_model), len(self.model), len(self.bagging_model))
        X_pred = np.array(X_pred)
        if len(self.model) >= 1:
            for _ in range(len(self.model)):
                print(X_pred.shape)
                for sn in range(len(self.model)):
                    test_pred = np.zeros((np.shape(X_pred)[0], len(self.model)))
                    gbm = self.stacking_model[_ * len(self.model) + sn]
                    try:
                        gbm.best_iteration
                        pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                    except:
                        pred = gbm.predict(X_pred)
                    test_pred[:, sn] = pred
                X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass
        # 普通cv
        for bn, gbm in enumerate(self.bagging_model):
            try:
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            except:
                pred = gbm.predict(X_pred)
            pred = np.array(pred)
            pred_out = pred if bn == 0 else pred_out + pred
        return pred_out / len(self.bagging_model)