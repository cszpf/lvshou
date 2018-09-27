# -*- coding: utf-8 -*-

from save_and_load import *
import numpy as np
import jieba
import warnings
warnings.filterwarnings('ignore')

data = load('./data/bidan_data')

word_list = '''

膨胀、水肿、癌症、绝症、心血管疾病、住院、畸形、冠心病、
长斑长痘、起疙瘩、晕倒、手术、脂肪三倍速度膨胀、毒素回流、
毒素反复吸收、回流心脏、压迫子宫、脂肪肝、糖尿病.等疾病、
等死、导致后果、危险、堵塞、肿瘤、脂肪凝固、脂肪回笼、毒素回笼、
萎缩、脂肪乱串、开刀、堵塞、花钱解决不了、游离、反增长、
长红疹、生活不能自理、体重快速上长、脂肪膨胀变大，
像滚雪球一样、导致、轻度脂肪肝变重度脂肪肝、水肿、像皮球一样
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
from sklearn.metrics import precision_score
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
fold = 100
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
print("f1:",f1_fold/fold)
print("recall:",recall_fold/fold)