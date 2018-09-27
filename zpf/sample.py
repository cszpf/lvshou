# 划分训练集与测试集，并对数据集打标签
# encoding=utf-8
import pandas as pd
import os
from divide import load_data
import numpy as np

def set_label(data, label, mode='test'):
    ''' 给数据打标签
    :param vec: object, could be CountVectorizer or TfidfTransformer
    :param mode: str, could be 'train' or 'test', default 'train'
        'train': train; 'test': test
    :return:
    '''
    data['ruleNameList'] = data['ruleNameList'].apply(str).apply(eval)
    data['correctResult'] = data['correctResult'].apply(str).apply(eval)
    index = []
    for i in data.index:
        x = data.loc[i]['ruleNameList']
        xx = data.loc[i]['correctResult']['correctResult']
        if label in x:
            # print(x,type(x), xx,type(xx), x.index(self.label))
            if xx[x.index(label)] == '1':
                index.append(i)
        elif label == '不违规':
            if '1' in xx:
                index.append(i)
    # _index = [(self.label in x.loc[i]) and (data['xx'].loc[i][x.loc[i].index(self.label)] == '1') for i in index]
    _data = data.loc[index]
    labels = [1] * len(index)
    not_data = data.drop(index)
    del (data)
    if mode == 'train':
        not_data = not_data.sample(len(_data.index))
    labels.extend([0] * len(not_data.index))
    data = pd.concat([_data, not_data])
    data['label'] = labels
    return data
if not os.path.exists('sample'):
    os.mkdir('sample')
_path = '../../data/Content'
dirs = os.listdir(_path)
for i in dirs:
    _trainpath = os.path.join(_path, i, '{}_train.csv'.format(i))
    _testpath = os.path.join(_path, i, '{}_test.csv'.format(i))
    train = load_data(_trainpath)
    tempnum = train.shape[0]
    train = set_label(train, i)
    assert train.shape[0] == tempnum
    if os.path.exists(_testpath):
        test = load_data(_testpath)
        test.drop(['content', 'mark_tag'], axis=1, inplace=True)
        test['label'] = 0
        train = pd.concat((train, test)).drop_duplicates('UUID')
        train.reset_index(inplace=True)
        train.drop('index', axis=1, inplace=True)
    uuids = np.array(train.index)
    for j in range(3):
        np.random.shuffle(uuids)
    train.iloc[uuids][['UUID', 'label']].\
        to_csv('sample/{}.csv'.format(i), index=False, encoding='utf-8')