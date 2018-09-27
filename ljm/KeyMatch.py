# encoding=utf-8
import sys
import time
import os
import pandas as pd
from project.interface import SupperModel
from project.divide import load_data
sys.path.append("..")

PATH = "../../data/Content"
SAMPLE_PATH = '../../data/Sample'
SETTING_PATH = "../setting"


def load_result(rule, result):
    path = os.path.join(PATH, rule, result)
    result = pd.read_csv(path, sep=',', encoding='utf-8')
    result.rename(columns={'result': 'label', 'pred': 'model_pred'}, inplace = True)
    return result[['UUID', 'label', 'model_pred']]


class KeyMatch(SupperModel):
    def __init__(self, rule, test_file, alone, **kags):
        self.rule = rule
        self.test_file = test_file
        self.key_words = []
        self.alone = alone
        self.data = None
        self.tokens = "transData.sentenceList"
        super(SupperModel, self).__init__()

    def load_key_words(self):
        with open(os.path.join(SETTING_PATH, self.rule + ".txt"), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.key_words.append(line.strip())

    def load_test(self):
        test_uuid = pd.read_csv(os.path.join(SAMPLE_PATH, self.test_file + ".txt"), header=None)
        rules = os.listdir(PATH)
        rules = [os.path.splitext(_)[0] for _ in rules]  # 所有违规标签
        if not self.alone:
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
        self.data = test_data[test_data['UUID'].isin(test_uuid.values[:, 0])]
        self.data.reset_index(drop=True, inplace=True)

    def match(self):
        self.load_key_words()
        self.load_test()
        label = []
        for tokens in self.data[self.tokens]:
            flag = 0
            for key_word in self.key_words:
                if key_word in tokens:
                    label.append(1)
                    flag = 1
                    break
            if not flag:
                label.append(0)
        self.data['key_pred'] = label

    def get_pred(self):
        return self.data[['UUID', 'key_pred']]

    def to_csv(self):
        self.data[['UUID', 'key_pred']].to_csv(os.path.join(PATH, self.rule, "key_match", self.test_file + ".txt"),
                                               sep=',', encoding='utf-8', index=False)


if __name__ == "__main__":
    rule = "敏感词"
    key_match = KeyMatch(rule, "sample1", alone=False)
    key_match.match()
    key_match.to_csv()
    key_pred = key_match.get_pred()
    model_pred = load_result(rule, "no/sample1_pred.csv")
    pred = model_pred.merge(key_pred, on="UUID")
    inter = []
    for i in range(len(pred)):
        if pred['model_pred'][i] == 1 and pred['key_pred'][i] == 1:
            inter.append(1)
        else:
            inter.append(0)
    print(sum(inter))
    print()
    key_match.acc(pred['label'], inter)
    print()
    key_match.acc(pred['label'], pred['key_pred'])
    print()
    key_match.acc(pred['label'], pred['model_pred'])
    print()
    # print(sum(pred['label']), sum(pred['model_pred']), sum(pred['key_pred']))
