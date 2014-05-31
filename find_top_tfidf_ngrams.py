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

docs_to_ngram_predictions = dict()


doc_count = 0
x_terms_to_predict = 90
words = []



ngram_counter = dict()
ngrams_by_doc = dict()


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

def getCorpusDocCountsForNGrams(n):
	global ngram_counter
	global corpusPath
	global doc_count
	global words
	global ngrams
	global ngrams_by_doc


	count = 1
	ngram_set = set()
	gl = glob.glob(corpusPath)
	print("There are " + str(len(gl)) + " files in the corpus")
	for filename in gl:
		
		#print "ngramming: " + filename + " -- " + str(count)
		words = (re.findall('\w+', open(filename).read().lower()))
		ngram_set.clear()
	

		ngrams = find_ngrams(words, n)

		
		ngrams_by_doc[filename] = ngrams

		for ngram in ngrams:
			if ngram not in ngram_set:
				ngram_set.add(ngram)

				if ngram not in ngram_counter:
					ngram_counter[ngram] = 0

				ngram_counter[ngram] += 1
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
		ngram_tf_idf_scores = calculate_tf_idf(filename)

		ngram_terms = get_x_top_terms(ngram_tf_idf_scores, x_terms_to_predict)
		docs_to_ngram_predictions[doc_id] = ngram_terms
		writeTopScorersToFile((filename.split(corpusPathRoot)[1][0:-4]), doc_id, x_terms_to_predict)

		

'''For each doc, calculates a tf-idf score for every word in the doc.Here tf-idf is  word_count * log((N -n)/n) where
word_count is the num occurances of the word in this doc, N is the total num docs, and rn is the num docs
in which this word occurs at least once. '''
def calculate_tf_idf(filename):

	global ngrams_by_doc
	global ngram_counter

	ngram_counts = collections.Counter(ngrams_by_doc[filename])

	total_ngrams_in_doc = sum(ngram_counts.values(), 0.0) #does not include ignored words


	ngram_score_tuples = []

	for ngram in list(set(ngram_counts.elements())):
		score = 0
		num_docs_ngram_appears_in = float(ngram_counter[ngram])
		idf_numerator = float(doc_count - num_docs_ngram_appears_in)
		if idf_numerator > 0:
			score = (ngram_counts[ngram]/ total_ngrams_in_doc) * math.log(idf_numerator/num_docs_ngram_appears_in) #tf * idf
		ngram_score_tuples.append([ngram, score])		

	return ngram_score_tuples


'''Returns the n highest scoring terms with their scores. '''
def get_x_top_terms(tf_idf_scores, x):
	sorted_score_tuples = sorted(tf_idf_scores, key=lambda pair: pair[1], reverse=True)
	return sorted_score_tuples[0:x]



def writeTopScorersToFile(file_id, doc_id, num_cands):
	path = printPath + file_id + ".cand"
	ensure_path(printPath)
	fh = open(path, "a")
	ngrams = docs_to_ngram_predictions[doc_id]
	for x in range (0, num_cands):
		ngram = ngrams[x]
		ngram_string = ' '.join(ngram[0])
		fh.write(ngram_string + "\n")
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
	printPath = "candidates/v270Cands/" + sys.argv[1] + "/"
	print("corpus path is: " + corpusPath)
	print("printing results to: " + printPath)
	for n in range(1, 4):
		print("Doing ngrams for n = " + str(n))
		getCorpusDocCountsForNGrams(n)
		processDocs()


if __name__ == "__main__":
	main()
