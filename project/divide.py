# encoding=utf-8
import pandas as pd
import os

PATH1 = '../../data/zhijian_data.csv'
PATH2 = '../../data/zhijian_data_20180709.csv'


def load_data(path=PATH1):
    try:
        with open(path, 'rb') as fr:
            data = pd.read_csv(fr, sep=',', encoding="utf-8")
    except:
        with open(path, 'rb') as fr1:
            data = pd.read_csv(fr1, sep=',', encoding="gbk")
    finally:
        pass
    try:
        data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval)\
            .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                              .replace("过度承诺效果问题", "过度承诺效果")
                              .replace("投诉倾向", "投诉")
                              .replace("提示客户录音或实物有法律效力", "提示通话有录音")
                              .replace("夸大产品功效", "夸大产品效果")for word in x])
        data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    except Exception:
        data.columns = ['UUID', 'relateData.sourceCustomerId', 'relateData.workNo', 'transData.sentenceList',
                        'manualData.isChecked', 'analysisData.isIllegal', 'analysisData.illegalHitData.ruleNameList',
                        'correctInfoData.correctResult', 'content', 'mark_tag']
        data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval) \
            .apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                   .replace("过度承诺效果问题", "过度承诺效果")
                   .replace("投诉倾向", "投诉")
                   .replace("提示客户录音或实物有法律效力", "提示通话有录音")
                   .replace("夸大产品功效", "夸大产品效果") for word in x])
        data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    return data


def divide_data(data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    all_rules = {}

    for i in range(len(data)):
        illegal_name = data['analysisData.illegalHitData.ruleNameList'][i]
        result = data['correctInfoData.correctResult'][i].get("correctResult")
        if '1' not in result:
            all_rules['不违规'] = all_rules.get('不违规', [])
            all_rules['不违规'].append(i)
            continue
        if len([0 for _ in result if _ =='1'])>1:
            all_rules['多类别'] = all_rules.get('多类别',[])
            all_rules['多类别'].append(i)
        for index, l in enumerate(illegal_name):
            if result[index] == '1':
                all_rules[l] = all_rules.get(l, [])
                all_rules[l].append(i)

    for _key, _value in all_rules.items():
        print(_key, len(_value))
        temp_data = data.iloc[_value]
        prepath = os.path.join(path, _key.replace('/', '-'))
        if not os.path.exists(prepath):
            os.makedirs(prepath)
        with open(os.path.join(prepath, '{}.csv'.format(_key.replace('/', '-'))), 'w') as fw:
            temp_data.to_csv(fw, sep=',', index=False, encoding='utf-8')


if __name__ == "__main__":
    data1 = load_data(PATH1)
    print(data1.shape)
    data2 = load_data(PATH2)
    print(data2.shape)
    data = pd.concat([data1,data2])
    print(data.shape)
    del(data2, data1)
    data.drop_duplicates(['UUID'], inplace=True)
    data.reset_index(inplace=True)
    print(data.shape)
    path = "../../data/Content/"
    divide_data(data, path)
