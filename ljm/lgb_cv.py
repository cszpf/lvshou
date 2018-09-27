import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import lightgbm as lgb

random.seed(2018)


def load_data(rule=''):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList']. \
        apply(eval).apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                          .replace("过度承诺效果问题", "过度承诺效果") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    if not rule:
        return data
    else:
        key_words = []
        counter = 0
        index = []
        with open(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                key_words.append(line.strip())

        # 对每个数据样本
        for sentences in data['agent_sentences']:

            # 针对该样本统计遍历违规词，计算是否在句子中
            for key_word in key_words:
                # 违规词在句子中
                if key_word in sentences:
                    index.append(counter)
                    break
            counter += 1
        return data.iloc[index].reset_index()


def lgb_cv(weight, label, k_fold, rule):
    train = weight
    kf = KFold(n_splits=k_fold)
    preds =[]
    clf = lgb.LGBMClassifier(num_leaves=35, max_depth=7, n_estimators=20000, n_jobs=20, learning_rate=0.01,
                             colsample_bytree=0.8, subsample=0.8)
    for train_idx, test_idx in kf.split(train):
        print(train_idx, test_idx)
        X = train[train_idx]
        y = label[train_idx]
        X_val = train[test_idx]
        y_val = label[test_idx]
        lgb_model = clf.fit(
            X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=100, verbose=1)
        test_preds = lgb_model.predict_proba(X_val)[:, 1]

        print("predicting...")
        preds.extend(test_preds)

    with open(r"E:\cike\lvshou\zhijian_data" + "\\" + rule + "\\" + "result\lgb_new.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')


def key_lgb_cv():
    train_weight = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\weight\train_weight.npy")
    train_label = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\weight\train_label.npy")
    test_weight = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\weight\val_weight.npy")
    test_label = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\weight\val_label.npy")
    test_key_label = np.load(r"E:\cike\lvshou\zhijian_data\敏感词\weight\key_match_val_label.npy")

    print(train_weight.shape)
    print(train_label.shape)
    print(test_weight.shape)
    print(test_label.shape)
    print(test_key_label.shape)

    pred = np.zeros(shape=test_label.shape, dtype=int)
    key_match_idx = []
    for i in range(len(test_key_label)):
        if test_key_label[i] == 1:
            key_match_idx.append(i)
    not_match_idx = [i for i in range(len(test_key_label)) if i not in key_match_idx]

    print("关键词匹配到的数据数量为：" + str(len(key_match_idx)))
    print("针对这些数据使用模型预测...")
    clf_val = lgb.LGBMClassifier(num_leaves=35, max_depth=7, n_estimators=20000, n_jobs=20, learning_rate=0.01,
                                 colsample_bytree=0.8, subsample=0.8)
    lgb_model_val = clf_val.fit(
        train_weight[:6000], train_label[:6000], eval_set=[(train_weight[:6000], train_label[:6000]),
                                                           (train_weight[6000:], train_label[6000:])],
        early_stopping_rounds=100, verbose=1)
    best_iter = lgb_model_val.best_iteration_
    print(best_iter)
    clf = lgb.LGBMClassifier(num_leaves=35, max_depth=7, n_estimators=best_iter, n_jobs=20, learning_rate=0.01,
                             colsample_bytree=0.8, subsample=0.8)
    lgb_model = clf.fit(
        train_weight, train_label, eval_set=[(train_weight, train_label)], early_stopping_rounds=100, verbose=1)

    key_match_pred = lgb_model.predict_proba(test_weight[key_match_idx])[:, 1]
    print(len(key_match_pred))
    key_match_label = []
    for p in key_match_pred:
        if p > 0.5:
            key_match_label.append(1)
        else:
            key_match_label.append(0)
    print(len(key_match_label))
    pred[key_match_idx] = key_match_label

    print("关键词匹配到的数据中：")
    print("precision: ", precision_score(test_label[key_match_idx], pred[key_match_idx]))
    print("recall: ", recall_score(test_label[key_match_idx], pred[key_match_idx]))
    print("accuracy: ", accuracy_score(test_label[key_match_idx], pred[key_match_idx]))
    print("micro :", f1_score(test_label[key_match_idx], pred[key_match_idx], average="micro"))
    print("macro: ", f1_score(test_label[key_match_idx], pred[key_match_idx], average="macro"))

    print("关键词未匹配到的数据中：")
    print("precision: ", precision_score(test_label[not_match_idx], pred[not_match_idx]))
    print("recall: ", recall_score(test_label[not_match_idx], pred[not_match_idx]))
    print("accuracy: ", accuracy_score(test_label[not_match_idx], pred[not_match_idx]))
    print("micro :", f1_score(test_label[not_match_idx], pred[not_match_idx], average="micro"))
    print("macro: ", f1_score(test_label[not_match_idx], pred[not_match_idx], average="macro"))

    print("全部数据中：")
    print("precision: ", precision_score(test_label, pred))
    print("recall: ", recall_score(test_label, pred))
    print("accuracy: ", accuracy_score(test_label, pred))
    print("micro :", f1_score(test_label, pred, average="micro"))
    print("macro: ", f1_score(test_label, pred, average="macro"))


def lgb_cv_k_fold(rule):
    # weight = np.load(r"E:\cike\lvshou\zhijian_data\count_weight_jjcw.npy")
    # label = np.load(r"E:\cike\lvshou\zhijian_data\label_jjcw.npy")
    # weight = np.load(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\count_weight_18000.npy")
    # label = np.load(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\label_18000.npy")
    weight = np.load(r"E:\cike\lvshou\data\Content\敏感词\敏感词_train_weight_only_sample1.npy")
    label = np.load(r"E:\cike\lvshou\data\Content\敏感词\敏感词_train_label_only_sample1.npy")
    print(weight.shape)
    print(label.shape)
    lgb_cv(weight, label, 5, rule)

    pred = []
    with open(r"E:\cike\lvshou\zhijian_data" + "\\" + rule + "\\" + "result\lgb_new.txt", 'r', encoding='utf-8') as f:
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


def show_result(rule, feature):
    label = np.load(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\label_" + str(feature) + ".npy")
    pred = []
    with open(r"E:\cike\lvshou\zhijian_data" + "\\" + rule + "\\" + "result\lgb_" + str(feature) + ".txt", 'r', encoding='utf-8') as f:
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
    lgb_cv_k_fold(rule="敏感词")
    # key_lgb_cv()
    # rules = ["过度承诺效果", "无中生有", "投诉倾向", "投诉", "服务态度生硬恶劣", "不礼貌", "草率销售", "违反指南销售"]
    # for rule in rules:
    #     lgb_cv_k_fold(rule=rule)

    # for rule in rules:
    #     print(rule)
    #     show_result(rule, feature=15000)
    #     print()
