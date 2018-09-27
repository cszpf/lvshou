import pandas as pd


def load_data():
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\conversation.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].apply(eval)
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    data['conversation'] = data['conversation'].apply(eval)
    return data


def divide(data, rule, path):
    all_rows_idx = []
    right_rows_idx = []
    wrong_rows_idx = []

    for i in range(len(data)):
        illegal_name = data['analysisData.illegalHitData.ruleNameList'][i]
        result = data['correctInfoData.correctResult'][i].get("correctResult")
        for index, l in enumerate(illegal_name):
            if l in rule:
                all_rows_idx.append(i)
                if result[index] == '1':
                    right_rows_idx.append(i)
                if result[index] == '2':
                    wrong_rows_idx.append(i)

    all_rows = data.iloc[all_rows_idx]
    right_rows = data.iloc[right_rows_idx]
    wrong_rows = data.iloc[wrong_rows_idx]

    all_rows.to_csv(path + "all.csv", sep=',', index=False, encoding='utf-8')
    right_rows.to_csv(path + "right.csv", sep=',', index=False, encoding='utf-8')
    wrong_rows.to_csv(path + "wrong.csv", sep=',', index=False, encoding='utf-8')


def get_agent_sentences(data):
    all_agent_sentences = []
    for conversation in data['conversation']:
        agent_sentences = []
        for sentence in conversation:
            if "AGENT" in sentence:
                agent_sentences.append(sentence['AGENT'])
        all_agent_sentences.append(' '.join(agent_sentences))
    data['agent_sentences'] = all_agent_sentences
    print(data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'agent_sentences']])
    data[['UUID', 'analysisData.illegalHitData.ruleNameList', 'correctInfoData.correctResult', 'agent_sentences']].\
        to_csv(r"E:\cike\lvshou\zhijian_data\agent_sentences.csv", sep=',', encoding='utf-8', index=False)


if __name__ == "__main__":
    data = load_data()
    # path = "E:\cike\lvshou\zhijian_data" + '\\'
    # divide(data, rule=['敏感词'], path=path + "敏感词" + '\\')
    # divide(data, rule=['禁忌称谓'], path=path + "禁忌称谓" + '\\')
    # divide(data, rule=['部门名称', '禁忌部门名称'], path=path + "禁忌部门名称" + '\\')
    get_agent_sentences(data)
