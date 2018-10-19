# encoding=utf-8
import pandas as pd
import jieba
from thulac import thulac
import os
from bdc import Featuers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import numpy as np
from sklearn import svm
import xgboost as xgb
import time
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

user_dict = 'setting/userdict2.txt'
stopwords = 'setting/stopwords'
data_path = '../../zhijian_data'
# the files' format is '逼单.csv' in content_path
content_path = '../data/content'
# the files' format is '逼单_test.csv' in sample_path
sample_path = '../data/sample'
# the files' format is 'a_token.csv' in token_path
token_path = '../../zhijian_data/tokens'

all_data = os.path.join(data_path, 'all_data.csv')
_thulac = thulac(user_dict, filt=1, seg_only=1)
with open(stopwords) as fr:
    stop_words = fr.read().replace('\n', ' ')

def tokenizer(string, mode='thulac'):
    if mode == 'thulac':
        return ' '.join([i[0] for i in _thulac.cut(string) if len(i[0]) > 1])
    elif mode == 'jieba':
        return ' '.join([i for i in jieba.cut(string) if i not in stop_words and len(i)>1])

def fenci(rule='部门名称', mode='thulac'):
    if mode == 'thulac':
        _rule_path = '{}/{}_token.csv'.format(token_path, rule)
    else:
        _rule_path = '{}/{}_{}_token.csv'.format(token_path, mode, rule)
    if os.path.exists(_rule_path):
        return pd.read_csv(_rule_path)
    if not os.path.exists(token_path):
        os.makedirs(token_path)
    data = pd.read_csv(all_data)
    content = pd.read_csv('{}/{}.csv'.format(content_path, rule))
    data = pd.merge(content, data, on='UUID')[['UUID', 'sentenceList', 'label']]
    data['sentenceList'] = data['sentenceList'].apply(str).apply(eval)\
        .apply(lambda x: [{'role': i['role'], 'content': tokenizer(i['content'], mode=mode)} for i in x])
    data.to_csv(_rule_path, index=False)
    return data

def genTrainData(rule='部门名称', mode='thulac', role='AGENT', feature_type='TFIDF',
                 ngram_range=(1, 3), _min=2, _max=0.9, _range=(0.4, 0.9), max_features=10000):
    print('This is the classification task on {}'.format(rule))
    data = fenci(rule, mode)
    if role not in 'AGENT USER':
        data['sentenceList'] = data['sentenceList'].apply(str).apply(eval) \
            .apply(lambda x: ' '.join([i['content'] for i in x]))
    else:
        data['sentenceList'] = data['sentenceList'].apply(str).apply(eval)\
        .apply(lambda x: ' '.join([i['content'] for i in x if i['role'] == role]))
    data['label'] = data['label'].apply(str).apply(eval)
    # data.columns = ['UUID','sentenceList','label']
    # test_data = pd.read_csv('{}/{}_test.csv'.format(sample_path, rule))
    test_data = data.tail(int(0.2*len(data.index)))
    _test = data.set_index('UUID').loc[test_data['UUID']]
    assert _test.shape[0] == test_data.shape[0]
    _train = data.set_index('UUID').drop(test_data['UUID'])
    assert _train.shape[0] + _test.shape[0] == data.shape[0]
    del(data, test_data)
    print('train/test:{}/{}'.format(_train.shape[0], _test.shape[0]))
    BDC_DF = select_Feature(_train['sentenceList'], _train['label'], _min=_min,
                            ngram_range=ngram_range, _max=_max, _range=_range, max_features=max_features)
    _vocab = {j: i for i, j in enumerate(BDC_DF.index)}
    del(BDC_DF)
    if _test.shape[0] == 0:
        return
    # _vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=3, max_features=int(0.6*len(_vocab.keys())))
    # _vec.fit(_train['sentenceList'])
    # _vocab = {j:i for i,j in enumerate(set(_vocab.keys()).union(set(_vec.vocabulary_)))}
    _vec = TfidfVectorizer(vocabulary=_vocab)
    _vec.fit(_train['sentenceList'])
    assert _vec.vocabulary_ == _vocab
    print('特征维度:', len(_vocab.keys()))
    train_csr = _vec.transform(_train['sentenceList'])
    test_csr = _vec.transform(_test['sentenceList'])
    _train.drop('sentenceList', axis=1, inplace=True)
    _test.drop('sentenceList', axis=1, inplace=True)
    _train = _train.reset_index().drop('index', axis=1)
    # model = CalibratedClassifierCV(svm.LinearSVC(random_state=2018))
    model = CalibratedClassifierCV(lgb.LGBMClassifier(metric='auc', learning_rate=0.02))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    models = []; y_preds, y_trues = [], []
    for train, test in skf.split(range(len(_train.index)), _train['label']):
        y_pred = 0
        model.fit(train_csr[train], _train.iloc[train]['label'])
        models.append(model)
        y_pred += np.array(model.predict_proba(train_csr[test]))[:, 1]
        y_preds.extend(np.array(y_pred).round()); y_trues.extend(_train.iloc[test]['label'])
    print('五折交叉的结果：')
    printMark(y_trues, y_preds)
    del(y_trues, y_preds, train, test)
    y_pred = 0
    for i in range(len(models)):
        y_pred += np.array(models[i].predict_proba(test_csr))[:, 1]/len(models)
    print('测试集的结果：')
    save_errorcase(_test.index, _test['label'], y_pred, rule)
    printMark(_test['label'], y_pred.round())


def printMark(y, y_pred):
    # print(y_pred)
    y, y_pred = np.array(y), np.array(y_pred)
    print('{}\t{}\t{}\t{}'.format('precision', 'recall', 'F1(micro)', 'F1(macro)'))
    print('{:f}\t{:f}\t{:f}\t{:f}'.format(precision_score(y, y_pred), recall_score(y, y_pred),
                                  f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro')))

def save_errorcase(uuids, y, y_pred, rule, state='test'):
    if not os.path.exists('setting/model'):
        os.makedirs('setting/model')
    # print(y, y_pred)
    assert np.shape(uuids)[0] == np.shape(y)[0]
    assert np.shape(uuids)[0] == np.shape(y_pred)[0]
    data = pd.DataFrame()
    data['UUID'] = uuids
    data['y_true'] = list(y); data['y_pred'] = y_pred
    # data['y_pred'] = data['y_pred'].apply(int)
    # data = data[(data['y_true'] ^ data['y_pred']) == 1]
    data.to_csv('setting/{}_{}.csv'.format(rule, state), index=False)
    print('save error case finish!')

def select_Feature(data, labels, ngram_range=(1, 3), _min=2, _max=0.9, _range=(0.4, 1.0),
                   max_features=10000):
    _Features = Featuers(k=80, ngram_range=ngram_range, _min=_min, _max=_max)
    df = _Features.calBdc(data, labels)
    df = df[(df['BDC'] >= _range[0]) & (df['BDC'] <= _range[1])]
    # return df[(df[1]>=1)&(df[0]==0)].sort_values(by=1).tail(max_features)
    return df

if __name__ == '__main__':
    start_time = time.time()
    dirs = [i.split('.')[0] for i in os.listdir(content_path)]
    for i in dirs:
        if i == '部门名称':
            genTrainData(rule=i, mode='thulac', role='AGENT', feature_type='TFIDF',
                         ngram_range=(1, 3), _min=3, _max=0.8, _range=(0.5, 1.0), max_features=30000)

    print('train cost {}s'.format(time.time()-start_time))