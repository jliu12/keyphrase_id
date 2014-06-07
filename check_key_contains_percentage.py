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


def inFile(length, line, unigrams, bigrams, trigrams):
	line = line.split()
	line = [x.lower() for x in line]
	if length == 1:
		format_line = line[0]
		if format_line in unigrams:
			return True
	elif length == 2:
		format_line = (line[0], line[1])
		if format_line in bigrams:
			return True
	elif length == 3:
		format_line = (line[0], line[1], line[2])
		if format_line in trigrams:
			return True

	return False

def print_percentage():
	#ensure_path (savePath)
	path = "percentages.info"
	write_file = open(path, "w")

	total_unfound= 0
	num_files = 0
	more_than_tri = 0
	num_key_words = 0

	for filename in glob.glob(corpusPath):
		num_files += 1
		write_file.write(filename + "\n")
		file_key_name = filename.split(".txt")[0] + ".key"

		key_file = open(file_key_name, 'r')

		words = (re.findall('\w+', open(filename).read().lower()))
		bigrams = find_ngrams(words, 2)
		trigrams = find_ngrams(words, 3)

		for line in key_file:
			num_key_words += 1
			length = len(line.split())
			if length <= 3:
				line = line.rstrip('\n')
				if not (inFile(length, line, words, bigrams, trigrams)):
					write_file.write(filename + ": " + line+"\n")
					total_unfound += 1
			else:
				more_than_tri += 1

	print "Percent unfound: " + str(float(total_unfound)/float(num_key_words))
	print "Percent larger than trigrams: " + str(float(more_than_tri) / float(num_key_words))
	write_file.close()


def main():
	global corpusPath
	global corpusPathRoot

	#savePath = "results/classifier_files/" + sys.argv[1] + "/"
	corpusPath = sys.argv[1] + "/*.txt"
	corpusPathRoot = sys.argv[1] + "/"

	print_percentage()


if __name__ == "__main__":
	main()