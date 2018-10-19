from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import pickle as pk
from collections import Counter
import os

# Bdc其实是一种有监督的词袋模型
class Featuers:
    
    def __init__(self, k=80, ngram_range=(1, 3), _min=2, _max=0.9, label=None, Bdc=True,
                 role='agent', debug=True, _maxfeature=50000):
        """初始化计算过程中的参数
        Parameters
        ----------
        k : int >= 1, default=80
            将稀疏矩阵转化成一般矩阵的速率
        ngram_range: tuple(min_n, max_n), default=(1,2)
            ngram的上下界,从文档中选择包含n个词的token,min_n<=n<=max_n

        label : string or int or None, default=None
            string or int表示的计算的是二分类的BDC值(依赖于你的类别标注),None表示计算的是多分类的BDC值
        Bdc : boolean, default=True
            True:采用论文中的公式计算二分类Bdc值,False:采用新公式计算BDC值
        role : string in ['agent', 'all', 'user'], default=agent
            'agent':只采用客服的对话,'all':采用所有的对话,'user':采用用户的对话
        debug:  boolean, default=True
            True:采用debug摸式,在debug模式下不允许从保存的中间文件中读取模型的中间结果
        """
        self.k, self.ngram = k, ngram_range
        self._min, self._max, self._maxfeature = _min, _max, _maxfeature
        self.label, self.Bdc, self.role, self.debug = label, Bdc, role, debug


    def getDF(self, data, labels):
        """ 将原始语料集矩阵表示
        Parameters
        ----------
        data : list like [string,...]
            语料集,每一单元是一篇文档.['This is the first document.','This is the second second document.',\
            'And the third one.','Is this the first document?',]
        labels : list or numpy.array, like [int,...]
            语料集对应的类别标注
        Returns
        -------
        df : pd.DataFrame
            token矩阵, index is token_id(int64), columns is label(string)
        vocab : dict
        token_id到token的映射词典, 形如{token_id:token,...}
        """
        vec = CountVectorizer(ngram_range=self.ngram, min_df=self._min, max_df=self._max)
        data = vec.fit_transform(data)
        vocab = {j: i for i, j in vec.vocabulary_.items()} # id2token
        
        _label = np.unique(labels)
        labels_token = {} # <dict>{str:[list]}
        k = self.k # 稀疏矩阵转化的速率
        for i in tqdm(range(0, len(labels), k)):
            if i+k >= len(labels):
                temp = labels[i:]
                temp_data = data[i:]
            else:
                temp = labels[i:i+k]
                temp_data = data[i:i+k]
            for _ in _label:
                labels_token[_] = labels_token.get(_, 0)
                labels_token[_] += temp_data[np.array(temp) == _].toarray().sum(axis=0)
        del(i, _, data, vec, labels, _label) # 防止内存泄露
        return pd.DataFrame(labels_token), vocab
    

    def calBdc(self, data, labels):
        """ 将原始语料集矩阵表示
        Parameters
        ----------
        data : list like [string,...]
            语料集,每一单元是一篇文档.['This is the first document.','This is the second second document.',\
            'And the third one.','Is this the first document?',]
        labels : list or numpy.array, like [int,...]
            语料集对应的类别标注
        Returns
        -------
        df : pd.DataFrame
            word_bdc矩阵, index is tokens, columns is ['TF','BDC']
        """
        if not os.path.exists('setting'):
            os.makedirs('setting')
        if not self.debug and os.path.exists('setting/{}data_vocab.pk'.format(self.role)):
            with open('setting/{}data_vocab.pk'.format(self.role), 'rb') as fr:
                df, vocab = pk.load(fr)
        else:
            df, vocab = self.getDF(data, labels)
            with open('setting/{}data_vocab.pk'.format(self.role), 'wb') as fw:
                pk.dump([df, vocab], fw)
        labels_counter = Counter(labels)
        label_list = [labels_counter[i] for i in df.columns]
        print('待计算bdc值的数据label为：', list(df.columns))
        # 扩展， 如果单纯的计算bdc可以将下列判断部分注释
        if self.label and self.Bdc: # 二分类bdc值
            label_list = [labels_counter[self.label], sum(labels_counter.values())]
            label_list[-1] -= label_list[0]
            df['negative'] = df.sum(axis=1) - df[self.label]
            df = df[[self.label, 'negative']]
        
        elif self.label and not self.Bdc: # 使用新公式计算二分类bdc值
            x = sum(labels_counter.values()) - labels_counter[self.label]
            label_list = [x if i != df.columns.index(self.label)\
             else j for i, j in enumerate(label_list)]
            
        assert len(label_list) == len(df.columns)

        # 计算Bdc
        temp_df = (df/label_list).apply(lambda x: x/sum(x), axis=1).applymap(lambda x: 0 if x==0 else x*np.log2(x))
        df['TF'] = df.sum(axis=1)
        df['BDC'] = round(temp_df.sum(axis=1)/np.log2(len(label_list)), 4) + 1
        df['Tokens'] = [vocab[i] for i in df.index]
        df.set_index(['Tokens'], inplace=True)
        df.to_csv('setting/{}_{}.csv'.format(self.role, self.label))
        return df


if __name__ == '__main__':
    # a samples
    corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?'
    ]
    labels = [0, 1, 2, 0]
    Bdc = Featuers()
    df = Bdc.calBdc(corpus, labels)
