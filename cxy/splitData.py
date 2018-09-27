# -*- coding: utf-8 -*-

from save_and_load import *
import pandas as pd
import json

right = []
wrong = []
check_term = '投诉'
name = 'tousu'
data = pd.read_csv('./data/%s.csv'%name,chunksize = 1000)
for part in data:
    for i in range(len(part)):
        line = part.iloc[i,:]
        content_id = line['UUID']
        result = line['correctInfoData.correctResult']
        result = result.replace("'",'"')
        result = json.loads(result)
        result = result['correctResult']
        
        chat = line['transData.sentenceList']
        chat = chat.replace("'",'"')
        chat = json.loads(chat)
        talk = []
        for sent in chat:
            talk.append((sent['role'],sent['content']))
        
        rule_list = line['analysisData.illegalHitData.ruleNameList']
        rule_list = rule_list.replace("'",'"')
        rule_list = json.loads(rule_list)
        if check_term not in rule_list:
            continue
        idx = rule_list.index(check_term)
        if result[idx]=='1':
            right.append((content_id,talk))
        else:
            wrong.append((content_id,talk))
            
#%%
word_list = '''
315、网上曝光、中央电视台、骗子、骗子公司、
权益、报警、消协、工商、药监、媒体、电视台、
记者、维权、投诉、法律程序、曝光、微博、
空间转发、告你们、报警、立案、维权、消协、搜集证据、
媒体、负面新闻、记者、恐吓、吓唬、花钱不减、咽不下这口气、
来处理、维权、采取措施、维权、警察、记者、打假、投诉315、
告你、投诉、打315、打官司、报警、投诉、打315、有录音、聊天记录、
负面消息、网络评论、准备来公司、申诉
'''

word_list = word_list.split('、')
word_list = list(set(word_list))

right_text = []
for chat in right:
    target = []
    talk = [line[0]+'\t'+line[1] for line in chat[1]]
    for i in range(len(talk)):
        line = talk[i]
        find_word = False
        for word in word_list:
            if word in line:
                find_word = True
                break
        if find_word:
            begin = max(0,i-3)
            end = min(i+4,len(talk))
            text = talk[begin:end]
            text = '\n'.join(text)
            for word in word_list:
                text = text.replace(word,'【'+word+'】')
            target.append(text)
            target.append('\n')
            
    if len(target)!=0:
        right_text.append(chat[0])
        right_text.extend(target)
        right_text.append('\n')

write(right_text,'./data/%s_right.txt'%name)
                

#%%
wrong_text = []
for chat in wrong:
    target = []
    talk = [line[0]+'\t'+line[1] for line in chat[1]]
    for i in range(len(talk)):
        line = talk[i]
        find_word = False
        for word in word_list:
            if word in line:
                find_word = True
                break
        if find_word:
            begin = max(0,i-3)
            end = min(i+4,len(talk))
            text = talk[begin:end]
            text = '\n'.join(text)
            for word in word_list:
                text = text.replace(word,'【'+word+'】')
            target.append(text)
            target.append('\n')
            
    if len(target)!=0:
        wrong_text.append(chat[0])
        wrong_text.extend(target)
        wrong_text.append('\n')

write(wrong_text,'./data/%s_wrong.txt'%name)
                