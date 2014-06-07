import re
import os
import sys
import os.path
import fnmatch
import glob
import collections
import json
import math
import nltk.data

from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip

#from feature_calc import FeatureCalculator

'''
So far, this file handles all txt documents in the corpusPath. 

'''
###
stopwords = []

corpusPath = "corpus/dummy/*.txt"
term_counter = collections.Counter()
docs_to_predictions = dict()
doc_count = 0
n_terms_to_predict = 8

def calcLDA(sentences):
	#stoplist = set('for a of the and to in the if it on'.split())
	cleanSentences = [[word for word in sentence.lower().split() if word not in stopwords]
		for sentence in sentences]
	allTokens = sum(cleanSentences, [])
	#tokensOnce = set(word for word in set(allTokens) if allTokens.count(word) == 1)
	#cleanSentences = [[word for word in sentence if word not in tokensOnce] for sentence in cleanSentences]

	id2word = corpora.Dictionary(cleanSentences)
	mm = [id2word.doc2bow(cleanSentence) for cleanSentence in cleanSentences]

	lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics = 3, update_every=1, chunksize=10, passes=100)

	for top in lda.print_topics():
		print top

	print 

'''Then processDocs goes through each txt file in the corpusPath again to find the n top-scoring
terms for that document. It stores a list of {doc_id: [{term1: score1}, {t2: s2}]}.'''
def processDocs():
	global docs_to_predictions
	for filename in glob.glob(corpusPath):
		doc_id = (filename[len("corpus/dummy/"):-4]) #drops ".txt" TODO: drop directory prefix. or not. This is for scoring purposes. 
		with open (filename) as f:
			text = f.read()
			tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			sentences = tokenizer.tokenize(text, False)
			calcLDA(sentences)
	
'''returns a counter with value of 1 for every word in the file. TODO: Could be improved!!! '''
def getWordsInTxtFile(filename):
	words = set(re.findall('\w+', open(filename).read().lower()))
	# From http://stackoverflow.com/questions/9082099/python-loop-through-files-in-a-folder-and-do-a-word-count
	ignore = ['the','a','if','in','it','of','or','on','and','to'] #'is', 'for', 'that']
	counter = collections.Counter(x for x in words if x not in ignore)
	return counter

def initStopWords():
	f = open('english')
	global stopwords
	stopwords = [line.strip() for line in open('english')]
	f.close()

def main():
	initStopWords()
	processDocs()
	#writeUnigramPredictions()

if __name__ == "__main__":
	main()
