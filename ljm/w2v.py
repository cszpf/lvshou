import jieba.posseg
import csv
import numpy as np
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def LR_fit(X, Y, T):
    """
    train and predict
    """
    print('fitting..')
    # print(Y)
    LR = LogisticRegression(C=1.0, max_iter=100, class_weight='balanced', random_state=8240, n_jobs=-1)
    LR.fit(X, Y)
    res = LR.predict(T)
    return res


def get_all_words(write_path, topk=20):
    writer = csv.writer(open(write_path, 'w', encoding='utf-8', newline=''))
    train_writer = csv.writer(open(r"E:\python\nlp\实体提取\data\w2v\train_words.csv", 'w', encoding='utf-8', newline=''))
    test_writer = csv.writer(open(r"E:\python\nlp\实体提取\data\w2v\test_words.csv", 'w', encoding='utf-8', newline=''))

    all_words = []
    train_reader = csv.reader(open(r"E:\python\nlp\实体提取\data\match.csv", 'r', encoding='utf-8'))
    for train_line in train_reader:
        body = train_line[1] + train_line[2]
        words = [word for word, pos in jieba.posseg.cut(body) if pos not in ['x']]
        all_words.append(words)
        u_words = set(words)
        word_num = {}
        for word in u_words:
            word_num[word] = words.count(word)
        ordered = sorted(word_num.items(), key=lambda d: d[1], reverse=True)
        '''if len(ordered) >= topk:
            data = [word[0] for word in ordered][:topk]
        else:
            data = [word[0] for word in ordered]'''
        data = [word[0] for word in ordered]
        train_writer.writerow([train_line[0], data])
        writer.writerow([train_line[0], words])

    test_reader = csv.reader(open(r"E:\python\nlp\实体提取\data\test_match.csv", 'r', encoding='utf-8'))
    for test_line in test_reader:
        body = test_line[1] + test_line[2]
        words = [word for word, pos in jieba.posseg.cut(body) if pos not in ['x']]
        all_words.append(words)
        u_words = set(words)
        word_num = {}
        for word in u_words:
            word_num[word] = words.count(word)
        ordered = sorted(word_num.items(), key=lambda d: d[1], reverse=True)
        '''if len(ordered) >= topk:
            data = [word[0] for word in ordered][:topk]
        else:
            data = [word[0] for word in ordered]'''
        data = [word[0] for word in ordered]
        test_writer.writerow([test_line[0], data])
        writer.writerow([test_line[0], words])
    return all_words


def load_vector_words(path):
    words_reader = csv.reader(open(path, 'r', encoding='utf-8'))
    all_words = []
    for row in words_reader:
        all_words.append(row[1])
    return all_words


def get_label(words_path, score_path):
    words_reader = csv.reader(open(words_path, 'r', encoding='utf-8'))
    score_reader = csv.reader(open(score_path, 'r', encoding='utf-8'))
    score_label_writer = csv.writer(open(r"E:\python\nlp\实体提取\data\w2v\score_label.csv", 'w', encoding='utf-8', newline=''))
    labels = []
    rows = []
    for row in score_reader:
        rows.append(row)
    for words_line in words_reader:
        for score_line in rows:
            if words_line[0] == score_line[0]:
                pos_num = int(score_line[2])
                neg_num = int(score_line[3])
                neu_num = int(score_line[4])
                if neg_num >= pos_num and neg_num >= neu_num:
                    labels.append(-1)
                elif pos_num >= neg_num and pos_num >= neu_num:
                    labels.append(1)
                else:
                    labels.append(0)
                score_label_writer.writerow([score_line[0], score_line[1], score_line[2], score_line[3],
                                             score_line[4], labels[-1]])
                break
    return labels


def train_model(all_words, size=100):
    print('正在训练w2v ...')
    print('size is: ', size)
    model = word2vec.Word2Vec(all_words, size=size, window=5, workers=4, min_count=1)
    savepath = r"E:\python\nlp\实体提取\data\w2v\model_" + str(size) + '.model'  # 保存model的路径
    print('训练完毕，已保存: ', savepath)
    model.save(savepath)


def generate_vector(words, size):
    print('载入模型中')
    model = word2vec.Word2Vec.load(r"E:\python\nlp\实体提取\data\w2v\model_"
                                   + str(size) + ".model")  # 填写你的路径
    print('加载成功')
    vec = np.zeros((len(words), size))
    for i, line in enumerate(words):
        counter = 0
        for word in line:
            try:
                vec[i] += np.array(model[word])
            except Exception as error:
                print(error)
        if counter != 0:
            vec[i] = vec[i] / float(counter)  # 求均值
    return vec


def validation(X, Y):
    """
    使用n-fold进行验证
    """
    print('validating...')
    fold_n = 3
    folds = KFold(n_splits=fold_n, random_state=0)
    score = np.zeros(fold_n)
    counter = 0
    for train_idx, test_idx in folds.split(X, Y):
        print(train_idx, test_idx)
        print(counter + 1, '-fold')
        X_train = X[train_idx]
        y_train = Y[train_idx]
        X_test = X[test_idx]
        y_test = Y[test_idx]
        res = LR_fit(X_train, y_train, X_test)
        fit_num = 0
        for index in range(len(y_test)):
            if y_test[index] == res[index]:
                fit_num += 1
        cur = fit_num * 1.0 / len(res)
        score[counter] = cur
        counter += 1
    print(score, score.mean())
    return score.mean()


if __name__ == '__main__':

    # choice = 'train'
    # choice = 'vector'
    choice = 'test'

    if choice == 'train':
        print("train model ... ")
        all_words = get_all_words(write_path=r"E:\python\nlp\实体提取\data\w2v\sentences.csv")
        train_model(all_words)

    if choice == 'vector':
        print("generator vector ... ")
        path = r"E:\python\nlp\实体提取\data\match.csv"

        train_words = load_vector_words(r"E:\python\nlp\实体提取\data\w2v\train_words.csv")
        test_words = load_vector_words(r"E:\python\nlp\实体提取\data\w2v\test_words.csv")
        train_vector = generate_vector(train_words, 100)
        test_vector = generate_vector(test_words, 100)
        np.save(r"E:\python\nlp\实体提取\data\w2v\train_vector_100", train_vector)
        np.save(r"E:\python\nlp\实体提取\data\w2v\test_vector_100", test_vector)

    if choice == 'test':
        print("k-fold validation ... ")
        print("载入向量中 ...")
        train_vector = np.load(r"E:\python\nlp\实体提取\data\w2v\train_vector_100.npy")
        print("载入成功")
        print(train_vector.shape)

        levels = get_label(words_path=r"E:\python\nlp\实体提取\data\w2v\train_words.csv",
                           score_path=r"E:\python\nlp\实体提取\data\score.csv")
        levels = np.array(levels)

        print(len(levels))
        res = validation(train_vector, levels)
        print('score: ', res)
