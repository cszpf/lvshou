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
        if not alone:
            if sentence.get("role") == "AGENT":
                sentence_content.append(sentence.get("content"))
        else:
            sentence_content.append(sentence.get("content"))
    return ' '.join(sentence_content)


class Tokens:
    def __init__(self, alone=False):
        '''
        :param alone:是否将user和agent的对话严格分开
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
            if not self.alone:
                file_name = _labels[i] + "_agent_sentences.csv" # XXX_sentences.csv
            else:
                file_name = _labels[i] + "_sentences.csv" # XXX_agent_sentences.csv
            _file_df.to_csv(os.path.join(prepath, file_name), sep=',',
                            encoding="utf-8", index=False)
            del _file_df
            # for _id in range(len(_file_df)):
            #     uuid = _file_df['UUID'][_id]
            #     sentenceList = _file_df['transData.sentenceList'][_id]
            #     if not self.alone:
            #         _contents = ['{}:{}'.format(_['role'], _['content']) for _ in sentenceList]
            #     else:
            #         _contents = ['{}'.format(_['content']) for _ in sentenceList if _['role'] == 'AGENT']
            #     contens = '\n'.join(_contents)
            #     save_file(contens, os.path.join(prepath, '{}-{}.txt'.format(uuid, _labels[i])))
        print('save all contents completed!')

    def makeToken(self):
        # thu1 = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注
        # thu1.cut_f("input.txt", "output.txt")  # 对input.txt文件内容进行分词，输出到output.txt
        jieba.load_userdict('../setting/userdict1.txt')

        _content_prepath = os.path.join(self.path, self.content) # ../../data/Content
        # _token_prepath = os.path.join(self.path, self.token)
        _files = os.listdir(_content_prepath)
        _files = [_ for _ in _files] # 所有的文件
        _labels = [os.path.splitext(_)[0] for _ in _files] # 所有违规标签

        for i, _file in enumerate(_files):
            print(i+1, _labels[i])
            # if not os.path.exists(_token_prepath):
            #     os.makedirs(_token_prepath)
            if not self.alone:
                file_name = _labels[i] + "_agent_sentences.csv"
                token_name = _labels[i] + "_agent_tokens.csv"
            else:
                file_name = _labels[i] + "_sentences.csv"
                token_name = _labels[i] + "_tokens.csv"
            data = load_data(os.path.join(_content_prepath, _labels[i], file_name))
            data['transData.sentenceList'] = data['transData.sentenceList'].\
                apply(lambda x: ' '.join([word for word in jieba.cut(x) if word not in [' ']]))
            data.to_csv(os.path.join(_content_prepath, _labels[i], token_name), sep=',',
                        encoding="utf-8", index=False)
            # _content_files = os.listdir(_content_prepath)
            # print(_content_files)
            # _content_files = [_ for _ in _content_files if '.txt' in _] # 所有的文件
            # print(_content_files)
            # for _ in _content_files:
            #     # print(_)
            #     # thu1.cut_f(os.path.join(_content_prepath, _), os.path.join(_token_prepath, _))
            #     fenci_file(os.path.join(_content_prepath, _), os.path.join(_token_prepath, _))
        print('Make Tokens of all files completed')


if __name__ == '__main__':
    start_time = time.time()
    _tokens = Tokens(alone=False)
    # print(get_stopwords())
    _tokens.makeContents()
    _tokens.makeToken()
    print(time.time()-start_time)
