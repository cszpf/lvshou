from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
# train model
def generateEmb(sentence):
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

def generateEmbGlove(sentence):
	if not os.path.exists('glove_model.b'):
		glove2word2vec(sentence, 'glove_model.b')
	model = keyedVectors.load_word2vec_format('glove_model.b')
	print(model)
	print(model.wv.vocab)
		

s = ['this is the first sentence for word2vec'.split(),
'this is the second sentence'.split(),
'yet another sentence'.split(),
'one more sentence'.split(),
'and the final sentence'.split()]
# generateEmb(s)
generateEmbGlove(s)