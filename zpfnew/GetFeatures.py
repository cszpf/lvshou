# encoding=utf-8
import pandas as pd
import os
import jieba
import thulac

user_dict = 'setting/userdict2.txt'
stopwords = 'setting/stopwords'
data_path = '../../zhijian_data'
content_path = '../data/content'
sample_path = '../data/sample'

path_type1 = 'zhijian_data_20180709.csv'
path_type2 = 'zhijian_data.csv'
path_type3 = '._content_marktag_2018{m:0>2}.csv'

with open(stopwords) as fr:
    stop_words = fr.read().split('\n')


jieba.load_userdict(user_dict)
_thulac = thulac.thulac(user_dict=user_dict, filt=True, seg_only=1)

def merge_word(word):
    return word.replace("禁忌部门名称", "部门名称")\
        .replace("过度承诺效果问题", "过度承诺效果")\
        .replace("投诉倾向", "投诉")\
        .replace("提示客户录音或实物有法律效力", "提示通话有录音")\
        .replace("夸大产品功效", "夸大产品效果")

def read_data(path):
    try:
        data = pd.read_csv(path, encoding='gbk')
    except Exception as e:
        data = pd.read_csv(path, encoding='utf-8')
    columns = data.columns
    data.columns = [i.split('.')[-1] for i in columns]
    data.drop(['sourceCustomerId', 'workNo', 'isIllegal', 'isChecked'], axis=1, inplace=True)
    data['ruleNameList'] = data['ruleNameList'].apply(str).apply(eval)\
        .apply(lambda x: [merge_word(word) for word in x])
    data['correctResult'] = data['correctResult'].apply(str).apply(eval)
    data['correctResult'] = data['correctResult'].apply(lambda x: x['correctResult'] if type(x) is dict else x)
    if 'mark_tag' in data.columns:
        for i in data.index:
            if data.iloc[i]['mark_tag'] == '质检错误':
                data.iloc[i]['correctResult'] = ['2'] * len(data.iloc[i]['ruleNameList'])
            elif data.iloc[i]['mark_tag'] == '转写错误':
                data.iloc[i]['correctResult'] = ['1'] * len(data.iloc[i]['ruleNameList'])
        data.drop(['content'], axis=1, inplace=True)
    else:
        data['mark_tag'] = None
    return data

def load_data(paths):
    data = pd.DataFrame()
    for i in paths:
        data = pd.concat((data, read_data(i)), sort=False)
        data = data.drop_duplicates('UUID').reset_index().drop('index', axis=1)
    return data

def split_data():
    paths = [path_type1, path_type2]
    paths.extend([path_type3.format(m=i) for i in range(7, 11)])
    data = load_data([os.path.join(data_path, i) for i in paths])
    data.to_csv(os.path.join(data_path, 'all_data.csv'), index=False)
    ruleNameList = {}
    for i in data.index:
        _data = data.iloc[i]
        for j in _data['ruleNameList']:
            ruleNameList[j] = ruleNameList.get(j, [])
            ruleNameList[j].append(i)
    _data = None
    for i in ruleNameList.keys():
        _data = data.iloc[ruleNameList[i], :]
        _label = []
        for j in _data.index:
            temp_data = data.iloc[j]
            if temp_data['mark_tag'] == '质检错误':
                _label.append(0)
            elif temp_data['mark_tag'] == '转写错误':
                _label.append(1)
            elif temp_data['correctResult'][temp_data['ruleNameList'].index(i)] == '1':
                _label.append(1)
            else:
                _label.append(0)

        _data['label'] = _label
        _data[['UUID', 'label']].to_csv('{}.csv'.format(os.path.join(content_path, i.replace('/', '-'))), index=False)
        _data = None

    print('split data finish!')

def fenci(string, mode='thulac'):
    if mode == 'thulac':
        return ' '.join([i[0] for i in _thulac.cut(string) if len(i[0]) > 1])
    elif mode == 'jieba':
        return ' '.join([i for i in jieba.cut(string) if i not in stop_words])

def get_Tokens():
    data = pd.read_csv(os.path.join(data_path, 'all_data.csv'))
    # data = data.head(50)
    data['sentenceList'] = data['sentenceList'].apply(str).apply(eval)\
        .apply(lambda x: [{'role': i['role'], 'content': fenci(i['content'], mode='thulac')} for i in x])
    data.to_csv(os.path.join(data_path, 'all_tokens.csv'), index=False)
    print('Tokenizer finish!')

def get_Sample():
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    dirs = os.listdir(content_path)
    for i in dirs:
        _data = pd.read_csv(os.path.join(content_path, i))
        _data.sample(int(0.2*_data.shape[0])).to_csv(os.path.join(sample_path, i), index=False)

if __name__ == '__main__':

    # data.to_csv('temp.txt', index=False)
    # print(data.columns)
    # print(data.head(50))
    # temp = data[data['ruleNameList'].apply(str).apply(eval).apply(lambda x: '部门名称' in x)]
    # print(temp.shape)
    # print(temp.head(20))
    # temp.to_csv('部门名称', index=False)
    if not os.path.exists(content_path):
        os.makedirs(content_path)
    # split_data()
    get_Tokens()


