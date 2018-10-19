# encoding=utf-8
import time
import pandas as pd
from GetFeatures import Exact
from model import Model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle as pk
import os
from scipy.sparse import csr_matrix
import re

PATH = '../../data/Content'
class Eval:
    def __init__(self, exact_mode='TF', _min=3, role='AGENT', label=None,
                 _ngram=(1, 1), features_type='TF', bdc_min=3, bdc_max=0.9,
                 select_min=0.0, select_max=1.0):
        '''建模训练
        :param exact_mode:提取特征的方式
        str or None, values in ['TF', 'DF'],
        default='TF': 利用正类中的TF挑选特征词, None表示不进行挑选
        :param _min:提取的特征需满足的最小值
        int,default=3:若exact_mode为'TF',则表示特征词在正类中的'TF'值需不小于_min
        :param role:提取指定role的对话文本
        str or None, None表示提取所有角色的对话文本， values in ['AGENT', 'USER'],
        default='AGENT': 只提取AGENT的对话文本
        :param label:进行二分类的类别标签
        str or None, default=None：表示对所有违规类别进行二分类
        str in ['禁忌称谓', '部门名称', '敏感词', '投诉', ...]
        :param _ngram: 计算BDC值所使用的ngram size
        tuple,default=(1,1),表示只使用uni_gram计算BDC值
        :param features_type: 特征值名称
        str, default='TF':表示使用TF值作为特征值,
        values in ['TF', 'TFBDC', 'TFIDF', 'TFIDFBDC', 'SUBTFIDF', 'SUBTFIDFBDC']
        :param bdc_min: 挑选出的特征词需满足的最小文档频率
        int or float, default=3 , float in [0,1]
        :param bdc_max: 挑选出的特征词需满足的最大文档频率
        int or float, default=0.9, float in [0,1]
        :param select_min: 挑选出的特征词需满足的最小BDC值
        float, default=0.0, float in [0,1]
        :param select_max: 挑选出的特征词需满足的最大BDC值
        float, default=1.0, float in [0,1]
        '''
        self.label = label
        self.Model = Model()
        self.Exact = Exact(exact_mode, _min, role, label, _ngram, bdc_min, bdc_max)
        self.features_type = features_type
        self.role = role
        self.select_min, self.select_max = select_min, select_max

    def generateData(self):
        dirs = os.listdir(PATH)
        if self.label:
            yield self.label
        else:
            for i in dirs:
                yield i

    def selectWords(self, string):
        # a = '零一二三四五六七八九十百千万两个位些上下前后左右中的地得些是喂话多少间么呢嘛因'
        # a = '一七的地得些这那'
        a = '一'
        pattern = re.compile(r'[{}]'.format(a), re.M)
        return len(re.findall(pattern, string))

    def train(self, label):
        train_path = '{a}/{b}/{b}_tokens_train.csv'
        test_path = 'sample/{}_test.csv'
        if os.path.exists(train_path.format(a=PATH, b=label)) \
                and os.path.exists(test_path.format(label)):
            data = pd.read_csv(train_path.format(a=PATH, b=label))
            # print('数据格式:', data.dtypes)
            if self.role in ['AGENT', 'USER']:
                data['sentenceList'] = data['sentenceList'].apply(str).apply(eval).\
                apply(lambda x: ' '.join([i['content'] for i in x if i['role'] == self.role]))
            else:
                data['sentenceList'] = data['sentenceList'].apply(str).apply(eval).apply(lambda x: ' '.join([i['content'] for i in x]))
            # data.columns = [UUID,sourceCustomerId,workNo,isIllegal,isChecked,sentenceList,label]
            # print(data['sentenceList'])
            test_sample = pd.read_csv(test_path.format(label))
            # test_sample.columns = ['UUID']
            # print(data.columns, test_sample.columns)
            data.set_index('UUID', inplace=True); test_sample.set_index('UUID', inplace=True)
            testdata = data.loc[test_sample.index]
            traindata = data.drop(test_sample.index)
            a = traindata.shape[0]
            b = traindata[traindata['label'] == 0].shape[0]
            c = testdata.shape[0]
            d = testdata[testdata['label'] == 0].shape[0]
            print('======={}=======\ntrain:{}:{}/{}\ntest:{}:{}/{}'.\
                  format(label, a, b, a-b, c, d, c-d))
            _df, _vocab = self.Exact.selectFeatures(traindata['sentenceList'], traindata['label'],
            _min=self.select_min, _max=self.select_max)
            del (data, test_sample)
            # _vocab1 = [i for i in _vocab.keys() if self.selectWords(i) == 0]
            # _vocab = {j:i for i,j in enumerate(_vocab1)}
            _vocab = {j: i for i, j in enumerate(_vocab.keys())}
            print('特征维度', len(_vocab.keys()))
            print('特征词', _vocab.keys())
            train_data, vec = self.generateVec(traindata['sentenceList'], _df, _vocab, _vec=None, mode=self.features_type)
            train_label = traindata[['label']]
            del(traindata)
            test_data, _vec = self.generateVec(testdata['sentenceList'], _df, _vocab, _vec=vec, mode=self.features_type)
            test_label = testdata[['label']]
            del(testdata)
            assert vec == _vec
            assert vec.vocabulary_ == _vocab
            gbms = self.Model.evl(train_data, train_label['label'], train_label.index,
                                  test_data, test_label['label'], test_label.index)
            self.saveModel(label, [_df, _vocab, vec, self.features_type, gbms])
            return
        print('======={}=======\n训练集或测试集缺失')
        return

    def generateVec(self, data, _df, _vocab=None, _vec=None, mode='TF'):
        if _vec:
            vec = _vec
            _data = vec.transform(data)
        else:
            if mode in ('TF', 'TFBDC'):
                vec = CountVectorizer(vocabulary=_vocab)
            elif mode in ('SUBTFIDF', 'SUBTFIDFBDC'):
                vec = TfidfVectorizer(vocabulary=_vocab, use_idf=1, smooth_idf=1, sublinear_tf=1)
            elif mode in ('TFIDF', 'TFIDFBDC'):
                vec = TfidfVectorizer(vocabulary=_vocab, use_idf=0, smooth_idf=0, sublinear_tf=0)
            _data = vec.fit_transform(data)

        if 'BDC' in mode:
            _data = _data.multiply(csr_matrix(_df.set_index('Tokens').loc[_vocab.keys()]['BDC']))

        assert {i: j for j, i in enumerate(_vocab.keys())} == _vocab
        assert _vocab == vec.vocabulary_
        return _data, vec

    def saveModel(self, label, model):
        with open('setting/{}.pk'.format(label), 'wb') as fw:
            pk.dump(model, fw)

    def loadModel(self, label):
        if not os.path.exists('setting/{}.pk'.format(label)):
            return

        with open('setting/{}.pk'.format(label), 'rb') as fr:
            return pk.load(fr)

    def predict(self):
        modelpath = 'setting/{}.pk'
        dirs = os.listdir(PATH)
        ruleNameList = []
        for i in dirs:
            model = self.loadModel(i)
            if model:
                # 分词+特征建模+模型预测
                pass


if __name__ == '__main__':
    eval1 = Eval(exact_mode='TF', _min=2, role='AGENT', label='部门名称',
                 _ngram=(1, 3), features_type='TF', bdc_min=1, bdc_max=0.9,
                 select_min=0, select_max=1.0)
    for i in eval1.generateData():
        eval1.train(i)