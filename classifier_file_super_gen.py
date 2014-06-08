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
import random
import errno
from itertools import tee, islice
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams


corpusPath = ""
#savePath = ""
corpusPathRoot = ""


def ensure_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def find_ngrams(input_list, n):
  	return zip(*[input_list[i:] for i in range(n)])


def getRandomGram(length, line, unigrams, bigrams, trigrams):
	neg_example = ""
	while True:
		if length == 1:
			neg_example = random.choice(unigrams)
		elif length == 2:
			neg_example = " ".join(random.choice(bigrams))
		elif length == 3:
			neg_example = " ".join(random.choice(trigrams))

		if neg_example != line:
			break

	#print "length: " + str(length) + ", pos: " + line + ", neg: " + neg_example
	return neg_example

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def writeToFile():
	#ensure_path (savePath)
	path = "train.info"
	write_file = open(path, "w")

	for filename in glob.glob(corpusPath):
		write_file.write(filename + "\n")
		file_key_name = filename.split(".txt")[0] + ".key"

		key_file = open(file_key_name, 'r')

		negative_examples = []

		words = (re.findall('\w+', open(filename).read().lower()))
		bigrams = find_ngrams(words, 2)
		trigrams = find_ngrams(words, 3)

		num_keys = file_len(file_key_name)

		for line in key_file:
			length = len(line.split())
			if length <= 3:
				line = line.rstrip('\n')
				write_file.write(line + ":1\n")
				neg_examples = getRandomGram(length, line, words, bigrams, trigrams)
				negative_examples.extend(neg_examples)

		for neg in negative_examples:
			write_file.write(neg + ":0\n")

		write_file.write("\n")

	write_file.close()


def main():
	global corpusPath
	global corpusPathRoot

	#savePath = "results/classifier_files/" + sys.argv[1] + "/"
	corpusPath = sys.argv[1] + "/*.txt"
	corpusPathRoot = sys.argv[1] + "/"

	writeToFile()


if __name__ == "__main__":
	main()