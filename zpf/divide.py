'''合并数据集，并按违规类型对数据集进行划分'''
# encoding=utf-8
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pickle as pk

PATH1 = '../../zhijian_data/zhijian_data.csv'
PATH2 = '../../zhijian_data/zhijian_data_20180709.csv'
TestPath1 = '../../zhijian_data/._content_marktag_201807.csv'
TestPath2 = '../../zhijian_data/._content_marktag_201808.csv'

def load_data(path=PATH1):
    '''
    从csv文件中读取数据，并按合并相似违规类型
    :param path:文件路径,str,default=PATH1
    :return:
    数据文件
    '''
    try:
        with open(path, 'rb') as fr:
            data = pd.read_csv(fr, encoding="utf-8")
    except:
        with open(path, 'rb') as fr1:
            data = pd.read_csv(fr1, encoding="gbk")
    finally:
        pass
    columns = data.columns
    data.columns = [i.split('.')[-1] for i in columns]
    data['ruleNameList'] = data['ruleNameList'].apply(eval)\
        .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                          .replace("过度承诺效果问题", "过度承诺效果")
                          .replace("投诉倾向", "投诉")
                          .replace("提示客户录音或实物有法律效力", "提示通话有录音")
                          .replace("夸大产品功效", "夸大产品效果")
               .replace('/', '-') for word in x])
    data['correctResult'] = data['correctResult'].apply(eval)
    return data

def divide_data(data, path, mode='train'):
    '''
    按违规类型对数据集进行划分
    :param data: 数据集,pd.DataFrame()
    :param path: 划分后的文件保存路径的父级目录
    :param mode: 区分数据集的类型,str,in['train','test'],default='train'
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    all_rules = {}

    for i in range(len(data)):
        illegal_name = data['ruleNameList'][i]
        for index, l in enumerate(illegal_name):
            all_rules[l] = all_rules.get(l, [])
            all_rules[l].append(i)

    for _key, _value in all_rules.items():
        print(_key, len(_value))
        temp_data = data.iloc[_value]
        prepath = os.path.join(path, _key.replace('/', '-'))
        if not os.path.exists(prepath):
            os.makedirs(prepath)
        with open(os.path.join(prepath, '{}_{}.csv'.format(_key.replace('/', '-'), mode)), 'w') as fw:
            temp_data.to_csv(fw, sep=',', index=False, encoding='utf-8')

def getAllSentence(dirs):
    '''
    合并数据集
    :param dirs:待合并的数据集的路径列表,list
    :return:
    合并之后的数据集:已对UUID去重
    '''
    data = load_data(dirs[0])
    print(data.shape)
    for i in range(1, len(dirs)):
        data1 = data.copy()
        data2 = load_data(dirs[i])
        print(data2.shape)
        data = pd.concat([data1, data2])
        print(data.shape)
        del(data2, data1)
    data.drop_duplicates(['UUID'], inplace=True)
    data.reset_index(inplace=True)
    if 'index' in data.columns:
        data.drop('index', axis=1, inplace=True)
    print(data.shape)
    return data

def getMainSentence(role, number=500):
    '''
    提取文本主干，跟实际应用无关
    :param role:
    :param number:
    :return:
    '''
    def fenci(x):
        '''
        对文本进行分词
        :param x: 文本
        :return:
        分词之后的文本列表
        '''
        return ' '.join([word for word in jieba.cut(x) if word not in (' ', '\n')])

    path1 = 'Tokens.csv'
    path2 = 'MainToken.csv'
    path3 = 'features.pk'
    if os.path.exists(path2):
        data = pd.read_csv(path2)
        with open(path3, 'rb') as fr:
            _vocab = pk.load(fr)
    else:
        if os.path.exists(path1):
            data = pd.read_csv(path1)
        else:
            jieba.load_userdict('setting/userdict1.txt')
            data = getAllSentence()
            data1 = data['sentenceList'].apply(str).apply(eval)
            data1 = data1.apply(lambda x: '\n'.join([i['content'] for i in x if i['role']==role]))
            data1 = data1.apply(lambda x: fenci(x))
            data['Tokens'] = data1
            del(data1)
            data.to_csv(path1, index=False)
        data = data.sample(4000)[['UUID', 'Tokens']]
        # print(list(data.sample(40)['Tokens']))
        # return
        # 提取特征词
        vec = CountVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.9, max_features=20000)
        vec.fit_transform(list(data['Tokens']))
        _vocab = [i for i in vec.vocabulary_.keys() if ' ' not in i]
        del(vec)
        print('已经训练好词表')
        # pattern = re.compile(r'{}'.format(' |'.join(_vocab)))
        data = data.sample(number)
        data.to_csv(path2, index=False)
        with open(path3, 'wb') as fw:
            pk.dump(_vocab, fw)
    data = data['Tokens']
    _data = []
    for i in list(data):
        temp = []
        for j in i.split(' '):
            if j in _vocab:
                temp.append(j)
        _data.append(' '.join(temp))
        temp = None
        # _data.append(''.join(pattern.findall(i)))
        i = None

    data = pd.DataFrame()
    data['MainToken'] = _data
    del(_data)
    data.to_csv('main.csv', index=False)
    # data['MainToken'] = data['Tokens'].apply(lambda x: ' '.join([i for i in x.split(' ') if i in _vocab]))
    # data.drop('Tokens', axis=1, inplace=True)


if __name__ == "__main__":
    data = getAllSentence([PATH1, PATH2])
    path = "../../data/Content/"
    divide_data(data, path, mode='train')
    data = getAllSentence([TestPath1, TestPath2])
    data = data[data['mark_tag'] == '质检错误']
    data.reset_index(inplace=True)
    if 'index' in data.columns:
        data.drop('index', axis=1, inplace=True)
    path = "../../data/Content/"
    divide_data(data, path, mode='test')
    # getMainSentence('AGENT')
