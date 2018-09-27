import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np

random.seed(2018)


jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\敏感词.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\部门名称.txt")
jieba.load_userdict(r"E:\cike\lvshou\zhijian_data\禁忌称谓.txt")


def load_data(rule=''):
    data = pd.read_csv(r"E:\cike\lvshou\zhijian_data\zhijian_data_20180709\data_cut.csv", sep=',', encoding="utf-8")
    data['analysisData.illegalHitData.ruleNameList'] = data['analysisData.illegalHitData.ruleNameList'].\
        apply(eval).apply(lambda x: [word.replace("禁忌部门名称", "部门名称")
                          .replace("过度承诺效果问题", "过度承诺效果") for word in x])
    data['correctInfoData.correctResult'] = data['correctInfoData.correctResult'].apply(eval)
    if not rule:
        return data
    else:
        key_words = []
        counter = 0
        index = []
        with open(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                key_words.append(line.strip())

        # 对每个数据样本
        for sentences in data['sentences']:

            # 针对该样本统计遍历违规词，计算是否在句子中
            for key_word in key_words:
                # 违规词在句子中
                if key_word in sentences:
                    index.append(counter)
                    break
            counter += 1
        return data, index


def cut_words(sentences):
    words = " ".join(list(jieba.cut(sentences)))
    return words


def count_vectorizer(data):
    # counter = {}
    # for line in data['words'].values:
    #     for word in line.split(' '):
    #         counter[word] = counter.get(word, 0) + 1
    #
    # counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    # with open(r"E:\cike\lvshou\zhijian_data\counter.txt", 'w', encoding='utf-8') as f:
    #     for key, value in counter:
    #         f.write(str(key) + ' : ' + str(value) + '\n')
    count_vect = CountVectorizer(max_df=0.5, min_df=3, max_features=10000)
    words_counter = count_vect.fit_transform(data['sentences'].values)
    weight = words_counter.toarray()

    # vectorizer = TfidfVectorizer(max_df=0.5, max_features=30000,
    #                              # stop_words=list(set(stopwords.words('english'))),
    #                              use_idf=True)
    # tfidf = vectorizer.fit_transform(data['words'].values)
    # weight = tfidf.toarray()
    return weight


def get_label_index(data, rule):
    index = []
    not_index = []

    # 对每个数据样本，遍历其检测出的违规类型
    for counter in range(len(data)):
        for i, item in enumerate(data['analysisData.illegalHitData.ruleNameList'][counter]):
            # 如果违规类型为要统计的类型且检测结果正确，总数量加1
            if rule == item and data['correctInfoData.correctResult'][counter].get("correctResult")[i] == '1':
                index.append(counter)
    for i in range(len(data)):
        if i not in index:
            not_index.append(i)
    return index, not_index


def get_count_weight():
    print("load data...")
    data = load_data()
    # rule = "敏感词"
    # data = load_data(rule=rule)
    # print("cutting...")
    # data['sentences'] = data['sentences'].apply(cut_words)
    print("count vectorizering...")
    weight = count_vectorizer(data)
    print("get label...")
    index, not_index = get_label_index(data, "敏感词")
    print(len(index), len(not_index))
    label = np.zeros(shape=(len(data, )), dtype=int)
    not_index = random.sample(not_index, len(index))
    label[index] = 1
    index.extend(not_index)
    random.shuffle(index)

    print("get weight and label...")
    weight = weight[index]
    label = label[index]

    print(weight.shape)
    print(label.shape)

    weight.dump(r"E:\cike\lvshou\zhijian_data\count_weight_mgc.npy")
    label.dump(r"E:\cike\lvshou\zhijian_data\label_mgc.npy")


def get_key_count_weight():
    print("load data...")
    rule = "敏感词"
    # 读入数据，获得关键词匹配到的数据的下标
    data, key_match_index = load_data(rule=rule)
    print(len(data))

    train_size = 7000
    test_size = 1000
    # print("cutting...")
    # data['sentences'] = data['sentences'].apply(cut_words)

    # 使用词袋模型，保存关键词匹配到的数据的weight
    print("count vectorizering...")
    weight = count_vectorizer(data)

    print("get label...")
    all_pos_index, all_neg_index = get_label_index(data, "敏感词")
    print("所有数据中，正样本数量为：" + str(len(all_pos_index)))
    print("所有数据中，负样本数量为：" + str(len(all_neg_index)))

    # 关键词匹配到的数据及特征权重
    key_match_data = data.iloc[key_match_index].reset_index()
    key_match_weight = weight[key_match_index]

    # 关键词未匹配到的数据及特征权重
    not_match_data = data.iloc[[i for i in range(len(data)) if i not in key_match_index]].reset_index()
    not_match_weight = weight[[i for i in range(len(data)) if i not in key_match_index]]

    # 在关键词匹配到的数据中获得正负样本的下标
    key_match_pos_index, key_match_neg_index = get_label_index(key_match_data, "敏感词")
    print("关键词匹配到的数据中，正样本数量为：" + str(len(key_match_pos_index)))
    print("关键词匹配到的数据中，负样本数量为：" + str(len(key_match_neg_index)))

    # 在关键词未匹配到的数据中获得正负样本的下标
    not_match_pos_index, not_match_neg_index = get_label_index(not_match_data, "敏感词")
    print("关键词未匹配到的数据中，正样本数量为：" + str(len(not_match_pos_index)))
    print("关键词未匹配到的数据中，负样本数量为：" + str(len(not_match_neg_index)))
    print()

    # 从关键词匹配到的数据中随机选取train_size条数据作为训练集正负样本
    pos_for_train = random.sample(key_match_pos_index, train_size)
    neg_for_train = random.sample(key_match_neg_index, train_size)

    # 剩余的关键词匹配到的数据可以选择作为验证集
    key_match_pos_for_val = random.sample([i for i in key_match_pos_index if i not in pos_for_train], test_size)
    key_match_neg_for_val = random.sample([i for i in key_match_neg_index if i not in neg_for_train],
                                          int(len(key_match_neg_index) / len(key_match_pos_index) * test_size))

    # 在关键词未匹配到的数据中选择数据作为验证集
    not_match_pos_for_val = random.sample(not_match_pos_index,
                                          int(len(not_match_pos_index) / len(key_match_pos_index) * test_size))
    not_match_neg_for_val = random.sample(not_match_neg_index,
                                          int(len(not_match_neg_index) / len(key_match_pos_index) * test_size))

    print("训练集中正样本数量为：" + str(len(pos_for_train)))
    print("训练集中负样本数量为：" + str(len(neg_for_train)))
    print()
    print("验证集关键词匹配到的数据中，正样本数量为：" + str(len(key_match_pos_for_val)))
    print("验证集关键词匹配到的数据中，负样本数量为：" + str(len(key_match_neg_for_val)))
    print("验证集关键词未匹配到的数据中，正样本数量为：" + str(len(not_match_pos_for_val)))
    print("验证集关键词未匹配到的数据中，负样本数量为：" + str(len(not_match_neg_for_val)))

    # 对应的label
    train_label = np.zeros(train_size * 2, dtype=int)
    val_label = np.zeros(len(key_match_pos_for_val) + len(not_match_pos_for_val) +
                         len(key_match_neg_for_val) + len(not_match_neg_for_val), dtype=int)

    val_key_match_label = np.zeros(len(key_match_pos_for_val) + len(not_match_pos_for_val) +
                                   len(key_match_neg_for_val) + len(not_match_neg_for_val), dtype=int)

    # 提取train_size * 2条训练集数据
    print("提取训练集...")
    train_weight = np.concatenate((key_match_weight[pos_for_train],
                                   key_match_weight[neg_for_train]), axis=0)
    train_label[:train_size] = 1

    # 按照数据集的原始比例提取验证集数据
    print("提取验证集...")
    val_weight = np.concatenate((key_match_weight[key_match_pos_for_val],
                                 not_match_weight[not_match_pos_for_val],
                                 key_match_weight[key_match_neg_for_val],
                                 not_match_weight[not_match_neg_for_val]), axis=0)

    val_label[:len(key_match_pos_for_val) + len(not_match_pos_for_val)] = 1
    val_key_match_label[:len(key_match_pos_for_val)] = 1
    val_key_match_label[len(key_match_pos_for_val) + len(not_match_pos_for_val):
    len(key_match_pos_for_val) + len(not_match_pos_for_val) + len(key_match_neg_for_val)] = 1

    print("训练集中，正样本数为：" + str(sum(train_label)))
    print("测试集中，正样本数为：" + str(sum(val_label)))
    print("测试集中，关键词匹配到的样本数为：" + str(sum(val_key_match_label)))

    train_idx = list(range(len(train_weight)))
    val_idx = list(range(len(val_weight)))
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    train_weight = train_weight[train_idx]
    train_label = train_label[train_idx]
    val_weight = val_weight[val_idx]
    val_label = val_label[val_idx]
    val_key_match_label = val_key_match_label[val_idx]

    train_weight.dump(r"E:\cike\lvshou\zhijian_data\敏感词\train_weight.npy")
    train_label.dump(r"E:\cike\lvshou\zhijian_data\敏感词\train_label.npy")
    val_weight.dump(r"E:\cike\lvshou\zhijian_data\敏感词\val_weight.npy")
    val_label.dump(r"E:\cike\lvshou\zhijian_data\敏感词\val_label.npy")
    val_key_match_label.dump(r"E:\cike\lvshou\zhijian_data\敏感词\key_match_val_label.npy")


def get_window_words(sentence, key_words, windows):
    words = []
    all_words = sentence.split(' ')
    index = [i for i, x in enumerate(all_words) if x in key_words]
    for i in index:
        begin = i - windows
        end = i + windows + 1
        if begin < 0:
            begin = 0
        if end > len(all_words):
            end = len(all_words)
        words.extend(all_words[begin:end])
    return ' '.join(words)


def get_key_window_weight(rule, feature, windows=0):
    print("load data...")
    data = load_data()
    if rule in ['敏感词', '部门名称', '禁忌称谓']:
        key_words = []
        with open(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                key_words.append(line.strip())

    if windows:
        data['sentences'] = data['sentences'].apply(lambda x: get_window_words(x, key_words, windows=windows))

    print("count vectorizering...")

    count_vect = CountVectorizer(max_df=0.5, min_df=3, max_features=feature)
    words_counter = count_vect.fit_transform(data['sentences'].values)
    weight = words_counter.toarray()
    print(weight.shape)

    print("get label...")
    rules = ["过度承诺效果", "无中生有", "投诉倾向", "投诉", "服务态度生硬/恶劣", "不礼貌", "草率销售", "违反指南销售"]
    for rule in rules:
        index, not_index = get_label_index(data, rule)
        label = np.zeros(shape=(len(data, )), dtype=int)
        not_index = random.sample(not_index, len(index))
        label[index] = 1
        index.extend(not_index)
        random.shuffle(index)
        print("get weight and label...")
        rule_weight = weight[index]
        rule_label = label[index]

        print(rule, rule_weight.shape)
        print(rule, rule_label.shape)

        if rule == "服务态度生硬/恶劣":
            rule = "服务态度生硬恶劣"
        rule_weight.dump(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\count_weight_" + str(feature) + ".npy")
        rule_label.dump(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\weight\label_" + str(feature) + ".npy")
        data['UUID'][index].to_csv(r"E:\cike\lvshou\zhijian_data" + '\\' + rule + "\\" + rule + "uuid.txt",
                                   index=False, encoding='utf-8')


if __name__ == "__main__":
    # get_key_count_weight()
    get_key_window_weight(rule="服务态度生硬/恶劣", feature=18000)
