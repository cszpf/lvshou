'''训练词向量'''
#encoding=utf-8
from gensim.models import Word2Vec
import os

def generateEmb(sentence):
	'''
	通过word2vec训练词向量
	:param sentence:文本列表,形如[['a','b','b'],...],'a b b' is the first sentence in corpus
	:return:
	词典(list),word2vec模型(object):假设模型为model,要获取词典中某个词x的向量表示，只需调用model[x]
	'''
	if os.path.exists('model.b'):
		model = Word2Vec.load('model.b')
		print(model)
		return list(model.wv.vocab), model
	else:
		model = Word2Vec(sentence, min_count=1)
		print(model)
		words = list(model.wv.vocab)
		print(words)
		print(len(model['this']))
		model.save('model.b')
		return words, model

if __name__ == '__main__':
	s = ['this is the first sentence for word2vec'.split(),
	'this is the second sentence'.split(),
	'yet another sentence'.split(),
	'one more sentence'.split(),
	'and the final sentence'.split()]
	generateEmb(s)