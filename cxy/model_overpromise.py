# -*- coding: utf-8 -*-

from save_and_load import *
import numpy as np
import jieba
import warnings
warnings.filterwarnings('ignore')

data = load('./data/overpromise_data')

word_list = '''满意为止、免费、100%、保证、绝对、书面协议、
法律效应、负责、效果为止、永不反弹、直到减下来为止、打包票、
不反弹、随便怎么、不会反弹、
根治、丰胸、治疗、治愈、癌症、排毒、增高、
检测结果，改善基因、乳腺增生
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
fold = 20
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