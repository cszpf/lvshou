import pandas as pd
import csv
import jieba
import os
import re
import numpy as np
from project.divide import load_data

PATH = "../../data/Content"
SETTING_PATH = "../setting"

mgc_key_words = []
with open(os.path.join(SETTING_PATH, "敏感词.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        mgc_key_words.append(line.strip())

bmmc_key_words = []
with open(os.path.join(SETTING_PATH, "部门名称.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        bmmc_key_words.append(line.strip())

jjcw_key_words = []
with open(os.path.join(SETTING_PATH, "禁忌称谓.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        jjcw_key_words.append(line.strip())


def error_data():
    path = os.path.join(PATH, '..', '..', 'label_error_data', 'label_error_data')
    error_type_counter = {}
    counter = 0
    for file in ['._content_marktag_201807.csv', '._content_marktag_201808.csv']:
        data = pd.read_csv(os.path.join(path, file), encoding='utf-8', sep=',')
        for tag in data['mark_tag']:
            counter += 1
            if tag in ['转写错误', '质检错误', '未知']:
                error_type_counter[tag] = error_type_counter.get(tag, 0) + 1
            else:
                error_type_counter['未标注'] = error_type_counter.get('未标注', 0) + 1
    print(counter)
    print(error_type_counter)


def get_content(sentence_list):
    sentence_list = eval(sentence_list)
    l = []
    for sentence in sentence_list:
        l.append({"role": sentence.get("role"), "content": sentence.get("content")})
    return l


def key_word_match(data):
    rule = data['ruleNameList']
    key_words = []
    if "敏感词" in rule:
        key_words.extend(mgc_key_words)
    if "部门名称" in rule:
        key_words.extend(bmmc_key_words)
    if "禁忌称谓" in rule:
        key_words.extend(jjcw_key_words)

    sentences = []
    for i in range(len(data["sentenceList"])):
        sentences.append(data["sentenceList"][i].get("content"))
    sentences = ' '.join(sentences)
    words = []
    for key in key_words:
        if key in sentences:
            words.append(key)
    return words


if __name__ == "__main__":
    # error_data()
    error_dir = os.path.join(PATH, '..', '..', 'label_error_data', 'label_error_data')
    files = ['._content_marktag_201808.csv']# , '._content_marktag_201808.csv']
    error_data = pd.DataFrame()
    for file in files:
        _ = pd.read_csv(os.path.join(error_dir, file), encoding='utf-8', sep=',')
        error_data = pd.concat([_, error_data], axis=0)
    error_data.reset_index(inplace=True, drop=True)
    error_data['ruleNameList'] = error_data['ruleNameList'].apply(eval)\
        .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
               .replace("过度承诺效果问题", "过度承诺效果")
               .replace("投诉倾向", "投诉")
               .replace("提示客户录音或实物有法律效力", "提示通话有录音")
               .replace("夸大产品功效", "夸大产品效果") for word in x])
    error_data['sentenceList'] = error_data['sentenceList'].apply(get_content)
    error_data['key_words'] = error_data.apply(key_word_match, axis=1)
    error_data.to_csv(os.path.join(error_dir, "._content_key_marktag_201808.csv"), encoding='utf-8', sep=',')