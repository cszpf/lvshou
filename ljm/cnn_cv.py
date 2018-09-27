import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping

random.seed(2018)


def load_data(rule=''):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].\
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


def cnn_cv(weight, label, k_fold):
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
        model.add(Convolution1D(64, 2, padding='same', input_shape=(10000, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Convolution1D(64, 2, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

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
    with open(r"E:\cike\lvshou\zhijian_data\cnn_pred.txt", 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p) + '\n')
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
        # model.save(r"..\cnn\level_top20_epoch_100_0.3_0.3.h5")
    return preds


def get_label_index(data, rule):
    index = []
    not_index = []

    # 对每个数据样本，遍历其检测出的违规类型
    for counter in range(len(data)):
        for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
            # 如果违规类型为要统计的类型且检测结果正确，总数量加1
            if rule == item and data['correctInfoData.correctResult'][counter].get("correctResult")[i] == '1':
                index.append(counter)
    for i in range(len(data)):
        if i not in index:
            not_index.append(i)
    return index, not_index


if __name__ == "__main__":
    weight = np.load(r"E:\cike\lvshou\zhijian_data\count_weight_mgc.npy")
    label = np.load(r"E:\cike\lvshou\zhijian_data\label_mgc.npy")

    cnn_cv(weight, label, k_fold=5)
    pred = []
    with open(r"E:\cike\lvshou\zhijian_data\cnn_pred.txt", 'r', encoding='utf-8') as f:
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