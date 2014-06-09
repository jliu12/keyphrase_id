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
candsPath = "candidates/v600Candsv2/v600Candsv2Train/"


def ensure_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def find_ngrams(input_list, n):
  	return zip(*[input_list[i:] for i in range(n)])


def getRandomGram(cands, keys):
	count = 0
	neg_list = []

	while True:
		neg_example = random.choice(cands)

		if neg_example not in keys:
			neg_list.append(neg_example)
			count += 1
			if (count == 3):
				break

	#print "length: " + str(length) + ", pos: " + line + ", neg: " + neg_example
	return neg_list

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def writeToFile():
	#ensure_path (savePath)
	path = "train.superinfo"
	write_file = open(path, "w")

	for filename in glob.glob(corpusPath):
		write_file.write(filename + "\n")
		file_key_name = filename.split(".txt")[0] + ".key"
		file_id_index = filename.rfind("/")
		file_id = (filename[file_id_index + 1:])[0:-4]

		print file_key_name
		print file_id_index
		print file_id

		cands_file_name = candsPath + file_id + ".cand"

		with open(cands_file_name) as f:
			cands_lines = f.read().splitlines()

		key_file = open(file_key_name, 'r')

		with open(file_key_name) as f:
			key_lines = f.read().splitlines()

		negative_examples = []

		num_keys = file_len(file_key_name)

		for line in key_file:
			length = len(line.split())
			if length <= 3:
				line = line.rstrip('\n')
				write_file.write(line + ":1\n")
				neg_examples = getRandomGram(cands_lines, key_lines)
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