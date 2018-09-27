import numpy as np
import random
import os
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import sys
import time
sys.path.append("..")
from project.divide import load_data, PATH1, PATH2
import lightgbm as lgb

random.seed(2018)

PATH = "../../data/Content"
SAMPLE_PATH = "../../data/Sample"


def get_label(data, rule):
    label = []
    # 对每个数据样本，遍历其检测出的违规类型
    for counter in range(len(data)):
        if rule not in data['analysisData.illegalHitData.ruleNameList'][counter]:
            label.append(0)
        else:
            for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
                if rule == item:
                    label.append(1 if data['correctInfoData.correctResult'][counter].
                                 get("correctResult")[i] == '1' else 0)
    return np.array(label)


def load_train_test(rule, test_file, only):
    uuid = pd.read_csv(os.path.join(SAMPLE_PATH, 'w2v', "uuid_all.txt"), sep=',', encoding="utf-8", header=None)
    vec = np.load(os.path.join(SAMPLE_PATH, 'w2v', "vec.npy"))
    train_path = os.path.join(PATH, rule, test_file[:-1], "no")
    if only:
        train_file_name = rule + "_train_only_" + test_file + ".csv"
        test_label = np.load(os.path.join(SAMPLE_PATH, rule, 'no', "test_label_only_" + test_file + ".npy"))
    else:
        train_file_name = rule + "_train_" + test_file + ".csv"
        test_label = np.load(os.path.join(SAMPLE_PATH, rule, 'no', "test_label_" + test_file + ".npy"))

    train_data = load_data(os.path.join(train_path, train_file_name))

    all_uuid = list(uuid.values[:, 0])
    train_uuid = list(train_data['UUID'].values)
    test_uuid = list(pd.read_csv(os.path.join(SAMPLE_PATH, test_file + ".txt"), header=None).values[:, 0])

    train_idx = []
    test_idx = []

    for uuid in train_uuid:
        train_idx.append(all_uuid.index(uuid))
    for uuid in test_uuid:
        test_idx.append(all_uuid.index(uuid))

    train_label = get_label(train_data, rule)
    return vec[train_idx], train_label, vec[test_idx], test_label


def lstm_cv(weight, label, k_fold):
    train = np.reshape(weight, (weight.shape[0], weight.shape[1], 1))
    print(train.shape)

    class_num = 2

    kf = KFold(n_splits=k_fold)
    preds = []
    pred_labels = []
    for train_idx, test_idx in kf.split(train):
        print(train_idx, test_idx)
        X = train[train_idx]
        y = label[train_idx]
        X_val = train[test_idx]
        y_val = label[test_idx]

        y = np_utils.to_categorical(y, class_num)
        y_val = np_utils.to_categorical(y_val, class_num)

        # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model = Sequential()

        # model.add(Dense(128, input_shape=(100, 1)))
        model.add(LSTM(64, input_shape=(300, 1)))
        model.add(Dense(class_num, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model.fit(X, y,
                  batch_size=16,
                  epochs=100,
                  verbose=1,
                  validation_data=(X_val, y_val), callbacks=[early_stopping])

        y_pred_label = model.predict(X_val)
        preds.extend(y_pred_label[:, 1])
        pred_labels.extend(np.argmax(y_pred_label, axis=1))
    with open(r"F:\cike\lvshou\data\Sample\w2v\lstm_pred.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
        # model.save(r"..\cnn\level_top20_epoch_100_0.3_0.3.h5")
    return preds


def lgb_cv(weight, label, k_fold):
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

    with open(r"F:\cike\lvshou\data\Sample\w2v\lgb_pred.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')


if __name__ == "__main__":
    train_weight, train_label, test_weight, test_label = load_train_test("敏感词", "sample1", only=False)
    lgb_cv(train_weight, train_label, k_fold=5)

    pred = []
    with open(r"F:\cike\lvshou\data\Sample\w2v\lgb_pred.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = float(line.strip())
            if p > 0.5:
                pred.append(1)
            else:
                pred.append(0)

    print("precision: ", precision_score(train_label, pred))
    print("recall: ", recall_score(train_label, pred))
    print("micro :", f1_score(train_label, pred, average="micro"))
    print("macro: ", f1_score(train_label, pred, average="macro"))
