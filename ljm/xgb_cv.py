import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import xgboost as xgb

random.seed(2018)


def xgb_cv(weight, label, k_fold, rule):
    train = weight

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 二分类的问题
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.05,  # 如同学习率
        'seed': 1000,
        'eval_metric': 'logloss',
    }
    num_rounds = 20000  # 迭代次数

    kf = KFold(n_splits=k_fold)
    preds = []
    for train_idx, test_idx in kf.split(train):
        print(train_idx, test_idx)
        X = train[train_idx]
        y = label[train_idx]
        X_val = train[test_idx]
        y_val = label[test_idx]

        xgb_train = xgb.DMatrix(X, label=y)
        xgb_val = xgb.DMatrix(X_val, label=y_val)

        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        print('Training xgboost model...')
        model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=False)
        print("best best_ntree_limit", model.best_ntree_limit)

        print("predicting...")
        test_preds = model.predict(xgb_val, ntree_limit=model.best_ntree_limit)
        preds.extend(test_preds)
    with open(r"E:\cike\lvshou\zhijian_data" + "\\" + rule + "\\" + r"result\xgb_15000.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')


def xgb_cv_k_fold():
    # weight = np.load(r"E:\cike\lvshou\zhijian_data\count_weight_jjcw.npy")
    # label = np.load(r"E:\cike\lvshou\zhijian_data\label_jjcw.npy")
    rules = ["过度承诺效果", "无中生有", "投诉倾向", "投诉", "服务态度生硬恶劣", "不礼貌", "草率销售", "违反指南销售"]
    for rule in rules:
        weight = np.load(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\count_weight_15000.npy")
        label = np.load(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\label_15000.npy")

        print(weight.shape)
        print(label.shape)
        xgb_cv(weight, label, 5, rule)

        pred = []
        with open(r"E:\cike\lvshou\zhijian_data" + "\\" + rule + "\\" + r"result\xgb_15000.txt", 'r',
                  encoding='utf-8') as f:
            for line in f.readlines():
                p = float(line.strip())
                if p > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)

        print("precision: ", precision_score(label, pred))
        print("recall: ", recall_score(label, pred))
        print("micro :", f1_score(label, pred, average="micro"))
        print("macro: ", f1_score(label, pred, average="macro"))


if __name__ == "__main__":
    xgb_cv_k_fold()
