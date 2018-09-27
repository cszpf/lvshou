import bdc
import numpy as np
import re
from train import Model
from divide import load_data
from tokens import get_sentences
from divide import getAllSentence
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import os
import jieba

PATH1 = '../../zhijian_data/zhijian_data.csv'
PATH2 = '../../zhijian_data/zhijian_data_20180709.csv'
TestPath1 = '../../zhijian_data/._content_marktag_201807.csv'
TestPath2 = '../../zhijian_data/._content_marktag_201808.csv'

class PreProcess:
    def __init__(self, label=None, role='agent', debug=1, tfbdc=1, ngram=(1, 2)):
        '''
        :param label: str, 待分类类别
        :param role: str, in ['agent', 'all', 'user'], default('agent')
            'agent'表示只抽取了agent的对话, 'all'表示抽取了全部的对话数据,'user'表示只抽取了用户的对话
        :param debug: bool, default True
            True表示采用开发者模式,即不直接从文件中读取中间结果
        '''
        self.label, self.role, self.debug = label, role, debug
        self.tfbdc = tfbdc
        self.mymodel = Model(label=label)

    def load_data(self, path):
        _labels = os.listdir(path)
        _files = [os.path.join(i, i+'_tokens.csv') for i in _labels]
        _files = [os.path.join(path, i) for i in _files]
        return _files, _labels

    def getBdc(self, data):
        ''' 得到bdc表
        :param path: 语料库路径
        :return: bdc表
        '''
        # print(data.shape)
        if not self.debug and os.path.exists('setting/{}_{}.csv'.format(self.role, self.label)):
            return pd.read_csv('setting/{}_{}.csv'.format(self.role, self.label), index_col=0)
        print('数据总量为', len(data['label']))
        return self.BDC.calBdc(data['sentenceList'], data['label'])

    def load_all_data(self):
        def clear_number(string):
            a = '零一二三四五六七八九十百千万两'
            pattern = re.compile(r'[{a}][{a}]+'.format(a=a), re.M)
            string = pattern.sub('', string)
            pattern = re.compile(r'[\d|a-z|A-Z]+', re.M)
            string = pattern.sub('', string)
            return ' '.join([word for word in jieba.cut(string) if word not in [' ', '\n']])
        jieba.load_userdict('setting/userdict2.txt')
        data1 = getAllSentence([PATH1, PATH2])
        data2 = getAllSentence([TestPath2, TestPath1])
        data2.drop(['content', 'mark_tag'], axis=1, inplace=True)
        data = pd.concat((data1, data2)).drop_duplicates('UUID')
        del(data1, data2)
        data.reset_index(inplace=True)
        if 'index' in data.columns:
            data.drop(['index'], axis=1, inplace=True)
        data['sentenceList'] = data['sentenceList'].apply(eval).\
            apply(lambda x: get_sentences(x, self.role)).\
            apply(lambda x: clear_number(x))
        if not os.path.exists('Data'):
            os.mkdir('Data')
        data[['UUID', 'ruleNameList', 'correctResult', 'sentenceList']].\
            to_csv('Data/all_data.csv', index=False)
        return data[['UUID', 'ruleNameList', 'correctResult', 'sentenceList']]

    def get_Features(self, mode, _min, _max, ngram_range, max_features, train_data, df_Bdc):
        if mode == 'DF':
            vec = CountVectorizer(min_df=_min, max_df=_max, ngram_range=ngram_range,
                                  max_features=max_features)
            # [train_data['label'] == 1]
            vec.fit_transform(train_data[train_data['label'] == 1]['sentenceList'])
            df_vocab = {i: j for j, i in vec.vocabulary_.items()}
            # df_Bdc = df_Bdc.loc[[df_vocab[i] for i in range(len(df_vocab.keys()))]].fillna(0)
            df_vocab = {i: j for j, i in df_vocab.items()}
            return df_vocab
        else:
            df_Bdc.sort_values(by=mode, inplace=True)
            _length = len(df_Bdc.index)
            if mode == 'BDC':
                if type(_max) is float:
                    df_Bdc = df_Bdc[df_Bdc[mode] <= _max]
                else:
                    df_Bdc = df_Bdc.head(_max)
                if type(_min) is float:
                    df_Bdc = df_Bdc[df_Bdc[mode] >= _min]
                else:
                    df_Bdc = df_Bdc.drop(df_Bdc.head(_min).index)
            else:
                if type(_min) is float:
                    df_Bdc = df_Bdc.drop(df_Bdc.head(int(_min * _length)).index)
                else:
                    df_Bdc = df_Bdc[df_Bdc[mode] >= _min]
                if type(_max) is int:
                    df_Bdc = df_Bdc[df_Bdc[mode] <= _max]
                else:
                    df_Bdc = df_Bdc.drop(df_Bdc.tail(int((1 - _max) * _length)).index)
            df_Bdc = df_Bdc.tail(max_features)
            _vocab = {j: i for i, j in enumerate(df_Bdc.index)}
            return _vocab

    def load_Dict(self, path='setting/userdict2.txt'):
        with open(path) as fr:
            data = fr.read().strip().split('\n')
        return {j: i for i, j in enumerate(data)}

    # 使用正则对长期信息进行建模
    def get_TF(self, dict, string):
        import re
        _data = []
        for i in dict.keys():
            pattern = re.compile(r'{}'.format(i), re.M)
            _data.append(len(pattern.findall(string)))
        return _data

    def get_XY1(self, sample_path='Sample',  ngram_range=(1, 1)):
        self.BDC = bdc.Featuers(label=1, role=self.role, debug=self.debug, ngram_range=ngram_range)
        def loaddata():
            path = 'Data/all_data.csv'
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                return self.load_all_data()
        _sample = os.listdir(sample_path)
        _sample = [os.path.join(sample_path, i) for i in _sample if 'sample' in i]
        AllData = loaddata()
        # print(AllData) 有输出
        for _ in _sample:
            test_uuid = pd.read_csv(_, header=None)
            test_uuid.columns = ['UUID']
            test_data = self.set_label(AllData[[i in list(test_uuid['UUID']) for i in AllData['UUID']]])
            # print(AllData[[i in list(test_uuid['UUID']) for i in AllData['UUID']]].shape)
            print('测试集数据规模:', test_data.shape)
            test_data.to_csv('./Data/test_data', index=False)
            # print(AllData.drop(test_data.index).shape)
            train_data = self.set_label(AllData.drop(test_data.index), mode='test')
            print('训练集数据规模:', train_data.shape)
            train_data.to_csv('./Data/train_data', index=False)
            _vocab = self.load_Dict()
            trn_csr = train_data['sentenceList'].apply(lambda x: self.get_TF(_vocab, x))
            trn_csr = np.array(list(trn_csr))
            print('特征维度', len(_vocab))
            print('训练集维度', len(trn_csr))
            # print(trn_csr)
            test_csr = test_data['sentenceList'].apply(lambda x: self.get_TF(_vocab, x))
            test_csr = np.array(list(test_csr))
            self.mymodel.evl(trn_csr, train_data['label'], train_data['UUID'],\
                              test_csr, test_data['label'], test_data['UUID'])
            del(trn_csr, train_data, test_data, test_csr, _vocab)
            # break

    def get_XY(self, sample_path='sample', mode='DF', _min=3, _max=0.9,
               max_features=30000, ngram_range=(1,2), **kgs):
        ''' 划分训练集\测试集
        :param sample_path: str, 测试集uuid路径
        :param mode: str, 特征提取方式, default('TF'), could be ('TF','BDC','DF')
            'tf':按照正样本的词频比例筛选,'df':按照词的出现文频,'bdc':按照bdc值
        :param _min: float or int, default(3), 最小阈值
            如果是float型, 排序之后按比例筛选; 如果是int型, 直接按出现频数筛选
        :param _max: float or int, default(0.9), 最大阈值
        :param max_features: int, default(10000), 最大特征数
        :return: 
        '''
        # 共用参数导致一个很大的问题
        self.BDC = bdc.Featuers(label=1, role=self.role, debug=self.debug,
                                ngram_range=ngram_range)
        def loaddata():
            path = 'Data/all_data.csv'
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                return self.load_all_data()
        _sample = os.listdir(sample_path)
        # _sample = [os.path.join(sample_path, i) for i in _sample if 'sample' in i]
        _sample = [os.path.join(sample_path, i) for i in _sample]
        AllData = loaddata()
        # print(AllData) 有输出
        for i, _ in enumerate(_sample):
            # print('=======================/nSample{}'.format(i+1))
            # test_uuid = pd.read_csv(_, header=None)
            # test_uuid.columns = ['UUID']
            # test_data = self.set_label(AllData[[i in list(test_uuid['UUID']) for i in AllData['UUID']]])
            # # print(AllData[[i in list(test_uuid['UUID']) for i in AllData['UUID']]].shape)
            # print('测试集数据规模:', test_data.shape)
            # test_data.to_csv('./Data/test_data', index=False)
            # # print(AllData.drop(test_data.index).shape)
            # train_data = self.set_label(AllData.drop(test_data.index), mode='test')
            # print('训练集数据规模:', train_data.shape)
            # train_data.to_csv('./Data/train_data', index=False)
            # if self.label not in _:
            #     continue
            print('===========\n%s'%_)
            all_data1 = pd.read_csv(_)
            train_data = pd.merge(all_data1.head(int(0.8*all_data1.shape[0])), AllData,
                                  how='left', on='UUID')
            # 欠采样
            train_data0 = train_data[train_data['label'] == 0]
            train_data1 = train_data[train_data['label'] == 1]
            if train_data0.shape[0] > train_data1.shape[0]:
                train_data0 = train_data0.sample(train_data1.shape[0])
                train_data = pd.concat((train_data0, train_data1))
            del(train_data1, train_data0)
            print('训练集数据规模:', train_data.shape)
            train_data.to_csv('./Data/train_data', index=False)
            test_data = pd.merge(all_data1.tail(int(0.2 * all_data1.shape[0])+1), AllData,
                                  how='left', on='UUID')
            print('测试集数据规模:', test_data.shape)
            test_data.to_csv('./Data/test_data', index=False)
            df_Bdc = self.getBdc(train_data)
            _vocab = self.get_Features(mode, _min, _max, ngram_range,
                                       max_features, train_data.copy(), df_Bdc.copy())
            df_vocab = _vocab.copy()
            if 'df_mode' in kgs.keys():
                df_mode, df_min, df_max = kgs['df_mode'], kgs['df_min'], kgs['df_max']
                df_vocab = self.get_Features(df_mode, df_min, df_max, ngram_range,
                                             max_features, train_data.copy(), df_Bdc.copy())
            df_Bdc = df_Bdc.loc[list(set(df_vocab.keys()) & set(_vocab.keys()))]
            _vocab = {j: i for i, j in enumerate(df_Bdc.index)}
            vec = TfidfVectorizer(ngram_range=ngram_range, vocabulary=_vocab, use_idf=1,
                                  smooth_idf=1, sublinear_tf=1)
            # vec = CountVectorizer(ngram_range=ngram_range, vocabulary=_vocab)
            trn_csr = vec.fit_transform(train_data['sentenceList'])
            assert _vocab == vec.vocabulary_
            print('特征维度', len(_vocab))
            test_csr = vec.transform(test_data['sentenceList'])

            # _vocab = self.load_Dict()
            # trn = train_data['transData.sentenceList'].apply(lambda x: self.get_TF(_vocab, x))
            # trn = np.array(list(trn))
            # # print(trn)
            # _test = test_data['transData.sentenceList'].apply(lambda x: self.get_TF(_vocab, x))
            # _test = np.array(list(_test))
            # print(trn_csr.toarray().shape)
            # print(trn.shape)
            # trn_csr = csr_matrix(np.concatenate((trn_csr.toarray(), trn), axis=1))
            # test_csr = csr_matrix(np.concatenate((test_csr.toarray(), _test), axis=1))
            # del(_vocab, trn, _test)
            del(_vocab, vec)
            if self.tfbdc:
                trn_csr = trn_csr.multiply(csr_matrix(df_Bdc['BDC']))
                test_csr = test_csr.multiply(csr_matrix(df_Bdc['BDC']))
            self.mymodel.evl(trn_csr, train_data['label'], train_data['UUID'],\
                              test_csr, test_data['label'], test_data['UUID'])
            del(trn_csr, train_data, test_data, test_csr, df_Bdc)
            # break


    def set_label(self, data, mode='test'):
        ''' 给数据打标签
        :param vec: object, could be CountVectorizer or TfidfTransformer
        :param mode: str, could be 'train' or 'test', default 'train'
            'train': train; 'test': test
        :return:
        '''
        data['ruleNameList'] = data['ruleNameList'].apply(eval)
        data['correctResult'] = data['correctResult'].apply(eval)
        index = []
        for i in data.index:
            x = data.loc[i]['ruleNameList']
            xx = data.loc[i]['correctResult']['correctResult']
            if self.label in x:
                # print(x,type(x), xx,type(xx), x.index(self.label))
                if xx[x.index(self.label)] == '1':
                    index.append(i)
            else:
                if '1' in xx:
                    index.append(i)
        # _index = [(self.label in x.loc[i]) and (data['xx'].loc[i][x.loc[i].index(self.label)] == '1') for i in index]
        _data = data.loc[index]
        labels = [1] * len(index)
        not_data = data.drop(index)
        del(data)
        if mode == 'train':
            not_data = not_data.sample(len(_data.index))
        labels.extend([0] * len(not_data.index))
        data = pd.concat([_data, not_data])
        data['label'] = labels
        return data

if __name__ == '__main__':
    P = PreProcess(role='agent', label='投诉', tfbdc=1, debug=1)
    # P.get_XY1()
    # P.getBdc()
    # df_mode = 'DF', df_min = 3, df_max = 0.5
    P.get_XY(ngram_range=(1, 3), max_features=20000, mode='BDC', _min=0.2, _max=0.95)