# encoding=utf-8
import jieba
import os
from divide import load_data
import time
import re

def save_file(data, path):
    with open(path, 'wb+') as fw:
        fw.write(data.encode('utf-8', 'ignore'))

# 返回停用词列表
def get_stopwords(path='../setting/stopwords.txt'):
    return []
    # stop_file = path
    # if not os.path.exists(stop_file):
    #     return []
    # with open(stop_file, 'rb+') as fr:
    #     data = fr.read()
    # stopwords = [_ for _ in data.decode('utf-8').strip().split('\n')]
    # del(data)
    # return stopwords

# 单个文件分词
# inFile, outFile为完整路径(unicode)
def fenci_file(inFile, outFile):
    # stopwords = get_stopwords()
    with open(inFile, 'rb') as fin:
        contens = fin.read().decode('utf-8')
    for _token in get_stopwords():
        contens = contens.replace(_token, "")
    for _token in get_stopwords('../setting/stopre.txt'):
        pattern = re.compile(r'%s' % _token)
        contens = pattern.sub('', contens)
    words = list(jieba.cut(contens, cut_all=False))
    # words = [word for word in words if len(word) > 1 and word != '\n']
    with open(outFile, "wb") as fout:
        fout.write(" ".join(words).encode('utf-8', 'ignore'))

def cut_words(inFile, outFile):
    pass

def get_sentences(sentence_list, alone):
    sentence_content = []
    for sentence in sentence_list:
        if alone == 'agent':
            if sentence.get("role") == "AGENT":
                sentence_content.append(sentence.get("content"))
        elif alone == 'user':
            if sentence.get("role") == "USER":
                sentence_content.append(sentence.get("content"))
        else:
            sentence_content.append(sentence.get("content"))
    return '\n'.join(sentence_content)


class Tokens:
    def __init__(self, alone='agent'):
        '''
        :param alone:str, default is 'agent', in ('agent', 'user', 'all')
            'agent': use agent's corpus; 'user': use user's corpus; 'all': use all corpus
        '''
        self.path = '../../data'
        self.content = 'Content'
        self.alone = alone

    def makeContents(self):
        _files = os.listdir(os.path.join(self.path, self.content))
        _files = [_ + '.csv' for _ in _files] # 所有的文件
        _labels = [os.path.splitext(_)[0] for _ in _files] # 所有违规标签
        print(_files)
        print(_labels)
        for i, _file in enumerate(_files):
            print(i+1, _labels[i])
            prepath = os.path.join(self.path, self.content, _labels[i]) # ../../data/Content/XXX
            _file_df = load_data(os.path.join(os.path.join(prepath, _file))) # ../../data/Content/XXX/XXX.csv
            if 'transData.sentenceList' not in _file_df.columns:
                continue
            _file_df['transData.sentenceList'] = _file_df['transData.sentenceList'].apply(eval)\
                .apply(lambda x: get_sentences(x, self.alone))
            file_name = _labels[i] + "_{}_sentences.csv".format(self.alone) # XXX_sentences.csv
            _file_df.to_csv(os.path.join(prepath, file_name), sep=',',
                            encoding="utf-8", index=False)
            del _file_df
        print('save all contents completed!')

    def makeToken(self):
        # thu1 = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注
        # thu1.cut_f("input.txt", "output.txt")  # 对input.txt文件内容进行分词，输出到output.txt
        # jieba.load_userdict('setting/userdict1.txt')

        _content_prepath = os.path.join(self.path, self.content) # ../../data/Content
        _files = os.listdir(_content_prepath)
        _files = [_ for _ in _files] # 所有的文件
        _labels = [os.path.splitext(_)[0] for _ in _files] # 所有违规标签

        for i, _file in enumerate(_files):
            print(i+1, _labels[i])
            # if not os.path.exists(_token_prepath):
            #     os.makedirs(_token_prepath)
            file_name = _labels[i] + "_{}_sentences.csv".format(self.alone)
            token_name = _labels[i] + "_{}_tokens.csv".format(self.alone)
            data = load_data(os.path.join(_content_prepath, _labels[i], file_name))
            data['transData.sentenceList'] = data['transData.sentenceList'].\
                apply(lambda x: ' '.join([word for word in jieba.cut(x) if word not in [' ']]))
            data.to_csv(os.path.join(_content_prepath, _labels[i], token_name), sep=',',
                        encoding="utf-8", index=False)
        print('Make Tokens of all files completed')


if __name__ == '__main__':
    start_time = time.time()
    _tokens = Tokens(alone='agent')
    # print(get_stopwords())
    _tokens.makeContents()
    _tokens.makeToken()
    print('本次处理总耗时', time.time()-start_time)
