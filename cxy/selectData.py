# -*- coding: utf-8 -*-
# 选择某一个类别的数据进行保存和分析
from save_and_load import *

class_data = []
check_term = '服务态度生硬/恶劣'
data = read('./data/zhijian_data.csv')
header = data.readline()
class_data.append(header.strip())
for line in data:
    if check_term in line:
        class_data.append(line.strip())
        
data = read('./data/zhijian_data_20180709.csv')
for line in data:
    if check_term in line:
        class_data.append(line.strip())
        
print(len(class_data))
write(class_data,'./data/badword.csv')