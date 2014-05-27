import re
import os
import sys
import getopt
import os.path
import fnmatch
import glob
import collections
import json
import math
import errno
from itertools import tee, islice
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams

'''
So far, this file handles all txt documents in the corpusPath. now tfidf_parser reports the n top scoring terms for each doc 
in the directory. tested only with dummy. Needs verification that calculation is actually working.
May need optimization to work on full corpus. Needs evaluation pipeline.

'''
###

corpusPath = ""
printPath = ""
corpusPathRoot = ""

docs_to_bigram_predictions = dict()
docs_to_trigram_predictions = dict()
#trigrams = []
#bigrams = []
doc_count = 0
n_terms_to_predict = 5
words = []

#bigram_counter = collections.Counter()
#trigram_counter = collections.Counter()

bigram_counter = dict()
trigram_counter = dict()
bigrams_by_doc = dict()
trigrams_by_doc = dict()

def ngramCounts(words, n):
	tlst = words
	while True:
		a, b = tee(tlst)
		l = tuple(islice(a, n))
		if len(l) == n:
			yield l
			next(b)
			tlst = b
		else:
			break

def getCorpusDocCountsForNGrams():
	global unigram_counter
	global bigram_counter
	global trigram_counter
	global corpusPath
	global doc_count
	global words
	global bigrams
	global trigrams
	global bigrams_by_doc
	global trigrams_by_doc

	count = 1
	bigram_set = set()
	trigram_set = set()
	gl = glob.glob(corpusPath)
	print("There are " + str(len(gl)) + " files in the corpus")
	for filename in gl:
		
		#print "ngramming: " + filename + " -- " + str(count)
		words = (re.findall('\w+', open(filename).read().lower()))
		bigram_set.clear()
		trigram_set.clear()

		bigrams = find_ngrams(words, 2)
		trigrams = find_ngrams(words, 3)
		
		bigrams_by_doc[filename] = bigrams
		trigrams_by_doc[filename] = trigrams

		for bigram in bigrams:
			if bigram not in bigram_set:
				bigram_set.add(bigram)

				if bigram not in bigram_counter:
					bigram_counter[bigram] = 0

				bigram_counter[bigram] += 1

		for trigram in trigrams:
			if trigram not in trigram_set:
				trigram_set.add(trigram)
				if trigram not in trigram_counter:
					trigram_counter[trigram] = 0
				trigram_counter[trigram] += 1

		doc_count += 1
		count += 1

	print "Finished corpus counts"

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

	
'''Then processDocs goes through each txt file in the corpusPath again to find the n top-scoring
terms for that document. It stores a list of {doc_id: [{term1: score1}, {t2: s2}]}.'''
def processDocs():
	global docs_to_predictions

	for filename in glob.glob(corpusPath):
		doc_id = (filename[0:-4]) #drops ".txt" TODO: drop directory prefix. or not. This is for scoring purposes. 
		#unigram_tf_idf_scores = calculate(tf_idf);
		bigram_tf_idf_scores, trigram_tf_idf_scores = calculate_tf_idf(filename)

		top_bigram_terms = get_n_top_terms(bigram_tf_idf_scores, n_terms_to_predict)
		top_trigram_terms = get_n_top_terms(trigram_tf_idf_scores, n_terms_to_predict)
		docs_to_bigram_predictions[doc_id] = top_bigram_terms
		docs_to_trigram_predictions[doc_id] = top_trigram_terms

		writeCountersToFile((filename.split(corpusPathRoot)[1][0:-4]), doc_id)

		

'''For each doc, calculates a tf-idf score for every word in the doc.Here tf-idf is  word_count * log((N -n)/n) where
word_count is the num occurances of the word in this doc, N is the total num docs, and n is the num docs
in which this word occurs at least once. '''
def calculate_tf_idf(filename):

	global bigrams_by_doc
	global trigrams_by_doc
	global bigram_counter
	global trigram_counter

	word_counts_bi = collections.Counter(bigrams_by_doc[filename])
	word_counts_tri = collections.Counter(trigrams_by_doc[filename])

	total_bigrams_in_doc = sum(word_counts_bi.values(), 0.0) #does not include ignored words
	total_trigrams_in_doc = sum(word_counts_tri.values(), 0.0)

	bigram_score_tuples = []
	trigram_score_tuples = []

	for bigram in list(set(word_counts_bi.elements())):
		score = 0
		num_docs_bigram_appears_in = float(bigram_counter[bigram])
		idf_numerator = float(doc_count - num_docs_bigram_appears_in)
		if idf_numerator > 0:
			score = (word_counts_bi[bigram]/ total_bigrams_in_doc) * math.log(idf_numerator/num_docs_bigram_appears_in) #tf * idf
		bigram_score_tuples.append([bigram, score])	

	for trigram in list(set(word_counts_tri.elements())):
		score = 0
		num_docs_trigram_appears_in = float(trigram_counter[trigram])
		idf_numerator = float(doc_count - num_docs_trigram_appears_in)
		if idf_numerator > 0:
			score = (word_counts_tri[trigram]/ total_trigrams_in_doc) * math.log(idf_numerator/num_docs_trigram_appears_in) #tf * idf
		trigram_score_tuples.append([trigram, score])	

	return bigram_score_tuples, trigram_score_tuples


'''Returns the n highest scoring terms with their scores. '''
def get_n_top_terms(tf_idf_scores, n):
	sorted_score_tuples = sorted(tf_idf_scores, key=lambda pair: pair[1], reverse=True)
	return sorted_score_tuples[0:n]


'''not currently used, just an experiment'''
def writeCountersToFile(file_id, doc_id):
	path = printPath + file_id + ".pred"
	ensure_path(printPath)
	fh = open(path, "w")
	bigrams = docs_to_bigram_predictions[doc_id]
	trigrams = docs_to_trigram_predictions[doc_id]

	for x in range (0, 5):
		bigram = bigrams[x]
		word_tuple = bigram[0]
		bigram_string = word_tuple[0] + " " + word_tuple[1]
		fh.write(bigram_string + "\n")


	for x in range (0, 5):
		trigram = trigrams[x]
		word_tuple = trigram[0]
		trigram_string = word_tuple[0] + " " + word_tuple[1] + " " + word_tuple[2]
		fh.write(trigram_string + "\n")

	fh.close()
  
def ensure_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# run as: python tdidf-ngrams.py [train/dev/test] directory_name 
def main():
	global corpusPath
	global printPath
	global corpusPathRoot

	corpusPath = sys.argv[2] + "/*.txt"
	corpusPathRoot = sys.argv[2] + "/"
	printPath = "results/v0/" + sys.argv[1] + "/"
	print("corpus path is: " + corpusPath)
	print("printing results to: " + printPath)
	getCorpusDocCountsForNGrams()
	processDocs()


if __name__ == "__main__":
	main()
