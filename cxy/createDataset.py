# -*- coding: utf-8 -*-
from save_and_load import *
import pandas as pd
import json

data = pd.read_csv('./data/badword.csv',iterator = 1000)

check_term = '服务态度生硬/恶劣'
data_set = []
right = 0
wrong = 0
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
            right+=1
            data_set.append((content_id,talk,1))
        else:
            wrong+=1
            data_set.append((content_id,talk,0))
            
save(data_set,'./data/badword_data')
print(right)
print(wrong)