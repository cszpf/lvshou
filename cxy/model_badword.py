# -*- coding: utf-8 -*-

from save_and_load import *
import numpy as np
import jieba
import warnings
from sklearn.metrics import precision_score
warnings.filterwarnings('ignore')

data = load('./data/badword_data')

word_list = '''
没素质、我操、草他妈、妈的、奇葩、有病、神经病、你麻痹、胖死你、
你他妈、我他妈、婊子、你大爷、我去年买了个表、你妹的、
妈了个逼的、你有病，神经、fuck、滚、你大爷、滚你老子，
你这个娘们、有病、坑货、有毛病、死胖子、狗日的、
狗儿子、滚、揣死你、干她、他妈的、我靠、你妹的、
你妈的逼、傻逼、傻叉、穷鬼、屌丝、肥婆、闭嘴、
你他妈的、操你妈、我靠、傻逼、你个老B、奇葩、
没教养、叼、胖死你、肥死你、
贱、有病、废话、叫穷、
永远别想、减下去、变态、不可理喻、
希望你搞清楚、爱瘦不瘦、毛线系、
跟你讲、耍我、你给我、闭嘴、毒死你、想太多、成熟一点、素质、骗你、
瞎说、不想跟你聊、凭什么、没有办法再跟你沟通下去、讲话真累，沟通不了、
好多问题，问这问那、不相信你可以不减呀、没有水平、法院传票、你这个鸟女人、
劈死你、胖女人、活该胖、会死么、
胖死、磨磨唧唧、找麻烦
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
print("f1:",f1_fold/fold)
print("recall:",recall_fold/fold)
