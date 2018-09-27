import pandas as pd
import csv
import jieba
import os
import re
import numpy as np
from project.divide import load_data

PATH = "../../data/Content"


def statistics(sample=False):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data_cut.csv", sep=',')
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval) \
        .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
               .replace("过度承诺效果问题", "过度承诺效果")
               .replace("投诉倾向", "投诉")
               .replace("提示客户录音或实物有法律效力", "提示通话有录音")
               .replace("夸大产品功效", "夸大产品效果") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)

    if sample:
        ids = pd.read_csv(r"E:\cike\lvshou\data\Sample\sample_proportion2.txt", header=None).values
        all_ids = data['UUID']
        indices = []
        for i, id in enumerate(all_ids):
            if id in ids:
                indices.append(i)
        data = data.loc[indices].reset_index()
        print(len(ids), len(data))
    all_illegal = []
    correct_illegal = []
    wrong_illegal = []
    counter = 0
    for illegals in data['analysisData.illegalHitData.ruleNameList']:
        result = data['correctInfoData.correctResult'][counter].get("correctResult")
        counter += 1
        for i, l in enumerate(illegals):
            if '1' in result[i]:
                correct_illegal.append(l)
            if '2' in result[i]:
                wrong_illegal.append(l)
            all_illegal.append(l)

    illegal_counter = {}
    correct_counter = {}
    wrong_counter = {}
    for illegal in set(all_illegal):
        illegal_counter[illegal] = all_illegal.count(illegal)
        correct_counter[illegal] = correct_illegal.count(illegal)
        wrong_counter[illegal] = wrong_illegal.count(illegal)
    print(len(illegal_counter), sorted(illegal_counter.items(), key=lambda x: x[1], reverse=True))
    print(len(correct_counter), sorted(correct_counter.items(), key=lambda x: x[1], reverse=True))
    # print(len(wrong_counter), sorted(wrong_counter.items(), key=lambda x: x[1], reverse=True))
    return dict(sorted(illegal_counter.items(), key=lambda x: x[1], reverse=True)), \
           dict(sorted(correct_counter.items(), key=lambda x: x[1], reverse=True))


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
        _ = load_data(os.path.join(PATH, rule, rule + suffix))
        test_data = pd.concat([test_data, _], axis=0)

    # 测试集样本空间
    test_data.drop_duplicates(['UUID'], inplace=True)
    test_data.reset_index(inplace=True)
    print(len(test_data))
    data = test_data[test_data['UUID'].isin(test_uuid.values[:, 0])]
    data.reset_index(drop=True, inplace=True)
    return data


def overlap_words(test_file, alone, rule, only):
    test_data = load_test(test_file, alone)
    if only:
        train_file = os.path.join(PATH, rule, test_file[:-1], rule + "_train_only_" + test_file + ".csv")
    else:
        train_file = os.path.join(PATH, rule, test_file[:-1], rule + "_train_" + test_file + ".csv")

    train_data = load_data(train_file)
    train_words = []
    test_words = []
    for words in train_data['transData.sentenceList']:
        train_words.extend(words.split(' '))

    for words in test_data['transData.sentenceList']:
        test_words.extend(words.split(' '))

    all_intersection_num = len(set(train_words).intersection(set(test_words)))
    all_train_num = len(set(train_words))
    all_test_num = len(set(test_words))
    print("训练集样本数：", len(train_data))
    print("测试集样本数：", len(test_data))
    print("训练集：", all_train_num)
    print("测试集：", all_test_num)
    print("交集：", all_intersection_num)
    print("共现词占训练集的比例：", all_intersection_num / all_train_num)
    print("共现词占测试集的比例：", all_intersection_num / all_test_num)


if __name__ == "__main__":
    # all_counter, corrent_counter = statistics()
    # all_sample_counter, corrent_sample_counter = statistics(sample=True)
    # all_sample_rate = {}
    # corrent_sample_rate = {}
    # for key, value in all_sample_counter.items():
    #     all_sample_rate[key] = "%.2f%%" % (value / all_counter.get(key, 0) * 100)
    # for key, value in corrent_sample_counter.items():
    #     if corrent_counter.get(key, 0) == 0:
    #         corrent_sample_rate[key] = 0
    #     else:
    #         corrent_sample_rate[key] = "%.2f%%" % (value / corrent_counter.get(key, 0) * 100)
    # print(len(all_sample_rate), all_sample_rate)
    # print(len(corrent_sample_rate), corrent_sample_rate)
    # print(len(all_sample_rate), sorted(all_sample_rate.items(), key=lambda x: x[1], reverse=True))
    # print(len(corrent_sample_rate), sorted(corrent_sample_rate.items(), key=lambda x: x[1], reverse=True))

    test_file = "sample_proportion"
    test_file = "sample"
    for i in range(5):
        print("禁忌称谓", test_file + str(i+1))
        overlap_words(test_file + str(i+1), alone=False, rule="禁忌称谓", only=True)
        print()

    test_file = "sample_proportion"
    for i in range(5):
        print("禁忌称谓", test_file + str(i+1))
        overlap_words(test_file + str(i+1), alone=False, rule="禁忌称谓", only=True)
        print()
