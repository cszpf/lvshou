# encoding=utf-8
import random
import pandas as pd
import os
import numpy as np
from divide import load_data, PATH1, PATH2

random_rate = 0.2
random.seed(2018)
PATH = "../../data/Sample"
PATH3 = '../../data/._content_marktag_201807.csv'
PATH4 = '../../data/._content_marktag_201808.csv'


def random_sample_data(data, file_name):
    length = data.shape[0]
    index = random.sample(range(length), int(float(length) * random_rate))
    print(len(index))
    uuid = data['UUID'][index]
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    uuid.to_csv(os.path.join(PATH, file_name), sep=',', encoding="utf-8", index=False)


def sample_data_proportion(data, file_name):
    all_rules = {}
    indices = []
    for i in range(len(data)):
        illegal_name = data['analysisData.illegalHitData.ruleNameList'][i]
        try:
            result = data['correctInfoData.correctResult'][i].get("correctResult")
        except Exception:
            result = data['correctInfoData.correctResult'][i]
        if '1' not in result:
            all_rules['不违规'] = all_rules.get('不违规', [])
            all_rules['不违规'].append(i)
            continue
        if len([0 for _ in result if _ == '1']) > 1:
            all_rules['多类别'] = all_rules.get('多类别', [])
            all_rules['多类别'].append(i)
            continue
        for index, l in enumerate(illegal_name):
            if result[index] == '1':
                all_rules[l] = all_rules.get(l, [])
                all_rules[l].append(i)
    for key, value in all_rules.items():
        indices.extend(random.sample(value, int(float(len(value) * random_rate)) + 1))
    indices = sorted(indices)
    print(len(indices))
    uuid = data['UUID'][indices]
    uuid.to_csv(os.path.join(PATH, file_name), sep=',', encoding="utf-8", index=False)


if __name__ == "__main__":
    data1 = load_data(PATH1)
    print(data1.shape)
    data2 = load_data(PATH2)
    print(data2.shape)
    data3 = load_data(PATH3)
    print(data3.shape)
    data4 = load_data(PATH4)
    print(data4.shape)
    data = pd.concat([data1, data2, data3, data4])
    print(data.shape)
    del(data1, data2, data3, data4)
    data.drop_duplicates(['UUID'], inplace=True)
    data.reset_index(inplace=True)
    print(data.shape)

    for i in range(5):
        random_sample_data(data, "sample_test" + str(i+1) + ".txt")
        # sample_data_proportion(data, "sample_proportion_all" + str(i+1) + ".txt")

        # test_uuid = pd.read_csv(os.path.join('../../data/Sample', "sample_proportion_all" + str(i + 1) + ".txt"), header=None)
        # counter = {}
        # il_counter = {}
        # data = data[data['UUID'].isin(test_uuid.values[:, 0])].reset_index()
        # _ = 0
        # for ruleNameList in data['analysisData.illegalHitData.ruleNameList']:
        #     try:
        #         result_list = data['correctInfoData.correctResult'][_].get("correctResult")
        #     except Exception:
        #         result_list = data['correctInfoData.correctResult'][_]
        #     for idx in range(len(ruleNameList)):
        #         counter[ruleNameList[idx]] = counter.get(ruleNameList[idx], 0) + 1
        #         if result_list[idx] == '1':
        #             il_counter[ruleNameList[idx]] = il_counter.get(ruleNameList[idx], 0) + 1
        #     _ += 1
        # print(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        # print(sorted(il_counter.items(), key=lambda x: x[1], reverse=True))
