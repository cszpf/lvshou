# -*- coding: utf-8 -*-

from save_and_load import *
import numpy as np
import jieba
import warnings
from sklearn.metrics import precision_score
warnings.filterwarnings('ignore')

data = load('./data/bidan_data')

word_list = '''
高利贷、贷款、借钱、信用卡、欠债、筹钱、外债、经济困难、欠钱、没钱、经济不允许、
公积金、返还现金、抵押、产权、上哪借、想不到办法、透支、压力大、资金困难、
逼迫、再办一张、不能自理、花呗、借呗、支付宝、蚂蚁花呗、
微信贷款、吃饭、借、离婚、为难、坑人、
产品被丢了、没收、银行卡、
吵架、矛盾、意见大、打架、开除，降级、最后一次、开除、降职、处罚、拉下老脸、
被公司、处罚、请财务、受委屈
'''
word_list = word_list.replace("\n","")
word_list = word_list.split('、')
word_list = list(set(word_list))

data_ = []
for line in data:
    id_ = line[0]
    label = line[2]
    text = ''
    for sent in line[1]:
        for word in word_list:
            if word in sent[1] and sent[0]=='AGENT':
                text+=sent[1]
                break
    data_.append((id_,text,label))
    

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score,recall_score
from tqdm import tqdm

def cut(x):
    x = jieba.lcut_for_search(x)
    return x

def evaluate(y,pred):
    y = np.array(y)
    pred = np.array(pred)
    acc = (y==pred).sum()/len(y)
    f1 = f1_score(y,pred)
    recall = recall_score(y,pred)
    p = precision_score(y,pred)
    #print("acc:",acc)
    #print("f1:",f1)
    return acc,f1,recall,p

data = [(line[1],line[2]) for line in data_]
data = np.array(data)

acc_fold = 0
f1_fold = 0
recall_fold = 0
p_fold = 0
fold = 40
for i in tqdm(range(fold)):
    order = np.random.permutation(len(data))
    train_rate = int(len(data)*0.9)
    train_index = order[0:train_rate]
    test_index = order[train_rate:]
    train = data[train_index]
    test = data[test_index]
    
    
    train_x = [line[0] for line in train]
    train_y = [int(line[1]) for line in train]
    test_x = [line[0] for line in test]
    test_y = [int(line[1]) for line in test]
    
    vec = TfidfVectorizer(ngram_range=(1,5),min_df=1, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1,tokenizer=cut)
    train_x = vec.fit_transform(train_x)
    test_x = vec.transform(test_x)
    
    from sklearn import svm
    model = svm.LinearSVC(C=5)
    model.fit(train_x,train_y)
    pred = model.predict(train_x)
    acc,f1,recall,p = evaluate(train_y,pred)
    pred = model.predict(test_x)
    acc,f1,recall,p = evaluate(test_y,pred)
    acc_fold+=acc
    f1_fold+=f1
    recall_fold += recall
    p_fold += p
print()
print("acc:",acc_fold/fold)
print("precision:",p_fold/fold)
print("recall:",recall_fold/fold)
print("f1:",f1_fold/fold)
