import re
import os
import sys
import os.path
import fnmatch
import glob
import collections
import json
import math

#from feature_calc import FeatureCalculator

'''
So far, this file handles all txt documents in the corpusPath. now tfidf_parser reports the n top scoring terms for each doc 
in the directory. tested only with dummy. Needs verification that calculation is actually working.
May need optimization to work on full corpus. Needs evaluation pipeline.

'''
###

corpusPath = "corpus/krapavin2000-train/*.txt"
term_counter = collections.Counter()
docs_to_predictions = dict()
doc_count = 0
n_terms_to_predict = 8

#counts the number of docs each word appears in at least once
'''getCorpusDocCountsForEachWord counts
the number of documents each word appears in at least once and stores these counts in term_counter. '''
def getCorpusDocCountsForEachWord():
	global term_counter #use global key word to modify global var
	global doc_count
	for filename in glob.glob(corpusPath):
		term_counter += getWordsInTxtFile(filename)
		doc_count += 1
	
'''Then processDocs goes through each txt file in the corpusPath again to find the n top-scoring
terms for that document. It stores a list of {doc_id: [{term1: score1}, {t2: s2}]}.'''
def processDocs():
	global docs_to_predictions
	for filename in glob.glob(corpusPath):
		doc_id = (filename[len("corpus/krapavin2000-train/"):-4]) #drops ".txt" TODO: drop directory prefix. or not. This is for scoring purposes. 
		tf_idf_scores = calculate_tf_idf(filename)
		top_terms = get_n_top_terms(filename, tf_idf_scores, n_terms_to_predict)
		docs_to_predictions[doc_id] = top_terms
		

'''For each doc, calculates a tf-idf score for every word in the doc.Here tf-idf is  word_count * log((N -n)/n) where
word_count is the num occurances of the word in this doc, N is the total num docs, and n is the num docs
in which this word occurs at least once. '''
def calculate_tf_idf(filename):
	word_counts = getWordCountForTxtFile(filename)
	total_words_in_doc = sum(word_counts.values(), 0.0) #does not include ignored words
	score_tuples = []
	for word in list(set(word_counts.elements())):
		score = 0
		num_docs_word_appears_in = float(term_counter[word])
		numerator = float(doc_count - num_docs_word_appears_in)
		if num_docs_word_appears_in < 1:
			print(word + " has n value of " + str(num_docs_word_appears_in))
		if numerator > 0:
			score = (word_counts[word]/ total_words_in_doc) * math.log(numerator/num_docs_word_appears_in) #tf * idf
		score_tuples.append([word, score])	
	return score_tuples


'''Returns the n highest scoring terms with their scores. '''
def get_n_top_terms(filename, tf_idf_scores, n):
	sorted_score_tuples = sorted(tf_idf_scores, key=lambda pair: pair[1], reverse=True)
	return sorted_score_tuples[0:n]

'''not currently used, just an experiment'''
def writeCounterToFile(ctr, path):
	jsonCtr = json.dumps(ctr)
	with open(path, 'w') as f:
		json.dump(jsonCtr, f)

#TODO: conversion from dict to counter has problems. The dict looks like it should, bu the
#ctr is missing  a lot of data. 
def testLoadCounterFromFile(path):
	data = collections.defaultdict()
	with open(path) as f:
		data = json.load(f)
	print(data)	
	ctr = collections.Counter(data)
        print(ctr)
	print("printed ctr that had been saved as json file")
	
'''returns a counter with value of 1 for every word in the file. TODO: Could be improved!!! '''
def getWordsInTxtFile(filename):
	words = set(re.findall('\w+', open(filename).read().lower()))
	# From http://stackoverflow.com/questions/9082099/python-loop-through-files-in-a-folder-and-do-a-word-count
	ignore = ['the','a','if','in','it','of','or','on','and','to'] #'is', 'for', 'that']
	counter = collections.Counter(x for x in words if x not in ignore)
	return counter


def getWordCountForTxtFile(filename):
	words = re.findall('\w+', open(filename).read().lower())
	# From http://stackoverflow.com/questions/9082099/python-loop-through-files-in-a-folder-and-do-a-word-count
	ignore = ['the','a','if','in','it','of','or','on','and','to'] #'is', 'for', 'that']
	counter = collections.Counter(x for x in words if x not in ignore)
	return counter

def writeUnigramPredictions():
	printPath = "results/vUnigram/train/"
	for docId, score_tuples in docs_to_predictions.iteritems():
		pathName = printPath + str(docId) + ".pred"
		#ensure_path(pathName)
		with open(pathName, 'w') as f:
			for t in score_tuples:
				f.write(t[0] + "\n")
				
def ensure_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise	
	
def main():
	#path = input("Enter file and path, place ' before and after the path: ")
	getCorpusDocCountsForEachWord()
	processDocs()
	writeUnigramPredictions()

if __name__ == "__main__":
	main()
