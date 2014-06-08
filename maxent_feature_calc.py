import nltk
import scipy
import numpy
import sys #MAXENT CHANGE
from nltk.classify import apply_features
from nltk.corpus import stopwords
from nltk.classify import maxent

#global variables for classification
stopwords = []


#outputPath = "v600TrainMaxentFullFeatureSet.txt"#MAXENT CHANGE
#candidatesPath = "v600CandsTrain.txt"#MAXENT CHANGE

outputPath = "v600TestProbsStrict.txt"#MAXENT CHANGE
candidatesPath = "test600cands.txt"#MAXENT CHANGE


class FeatureCalculator:
	def __init__(self, document, phrases, tfidf):
		self.phrases = phrases
		self.tfidf = self.read_tfidf(tfidf) # this is a dict!!!
		self.document = self.read_doc(document)

	def read_doc(self, f_name):
		with open (f_name, "r") as new_file:
			data = new_file.read().replace('\n', '').lower()
			return data

	def read_tfidf(self, tfidf):
		new_dict = {}
		i = 0
		for s in self.phrases:
			new_dict.update({s[0]: tfidf[i]})
			i += 1
		return new_dict

	#Measures the length of the keyphrase
	#
	def ft_keyphrase_len(self, candidate):
		return {'length': len(candidate)}

	#part of speech tagging
	def ft_pos(self, candidate):
		#not yet implemented, how to represent POS as a value?
		return {'pos': 1}

	#how far into the document the candidate first appears. 
	#doc is represented as a string
	def ft_first_occurrence_position(self, candidate):
		relative_pos = float(self.document.find(candidate)) / len(self.document)
		return {'first_occur': relative_pos}

	def ft_last_occurrence_position(self, candidate):
		relative_pos = float(self.document.rfind(candidate)) / len(self.document)
		return {'last_occur': relative_pos}

	def ft_spread(self, start, end):
		return {'spread': end - start}

	def ft_num_tokens(self, candidate):
		num_tokens = len(candidate.split())
		return {'num_tokens': num_tokens}

	#TODO: implement
	def ft_doc_section(self, candidate):
		#not yet implemented, need document section markers
		return {'section': 1}

	def ft_tagger(self, candidate):
		tokens = nltk.word_tokenize(candidate)
		tag_list = nltk.pos_tag(tokens)
		tag_string = ""
		for s in tag_list:
			if (tag_string != ""):
				tag_string += " "
			tag_string += s[1]
		return {'tags': tag_string}

	def ft_tfidf(self, candidate):
		return{'tfidf': self.tfidf[candidate]}

	def ft_stopword_start(self, candidate):
		first_word = candidate.partition(' ')[0]
		is_stopword = 0
		if first_word in stopwords:
			is_stopword = 1
		return {'stopword_start': is_stopword}

	def ft_stopword_end(self, candidate):
		last_word = candidate.partition(' ')[-1]
		is_stopword = 0
		if last_word in stopwords:
			is_stopword = 1
		return {'stopword_end': is_stopword}

	def get_phrases(self):
		return self.phrases 

	#calculates a dictionary of all the features for a word
	def get_features(self, candidate):
		keyphrase_len = self.ft_keyphrase_len(candidate)
		feature_dict = keyphrase_len

		first_occurrence = self.ft_first_occurrence_position(candidate)
		feature_dict.update(first_occurrence)

		last_occurrence = self.ft_last_occurrence_position(candidate)
		feature_dict.update(last_occurrence)

		spread = self.ft_spread(first_occurrence['first_occur'], last_occurrence['last_occur'])
		feature_dict.update(spread)

		num_tokens = self.ft_num_tokens(candidate)
		feature_dict.update(num_tokens)

		part_of_speech = self.ft_tagger(candidate)
		feature_dict.update(part_of_speech)

		tfidf = self.ft_tfidf(candidate)
		feature_dict.update(tfidf)

		stopword_start = self.ft_stopword_start(candidate)
		feature_dict.update(stopword_start)

		stopword_end = self.ft_stopword_end(candidate)
		feature_dict.update(stopword_end)
		
		return feature_dict

def init_global_vars():
	# init stopwords
	f = open('english')
	global stopwords
	stopwords = [line.strip() for line in open('english')]
	f.close()

def parse_train_data(fname):
	data = {}
	new_file = True
	new_filename = ""
	new_list = []
	with open(fname, "r") as f:
		for line in f:
			if new_file:
				new_filename = line.strip()
				new_file = False
			elif len(line) == 1:
				#store last file and reset vars
				data[new_filename] = new_list
				new_filename = ""
				new_list = []
				new_file = True
			else:
				words = line.strip().split(':')
				example = "no"
				if words[1] == "1":
					example = "yes"
				new_tup = (words[0], example)
				new_list.append(new_tup)
	if new_filename != "":
		data[new_filename] = new_list
	return data

def parse_test_data(fname):
	data = {}
	new_file = True
	new_filename = ""
	new_list = []
	with open(fname, "r") as f:
		for line in f:
			if new_file:
				new_filename = line.strip()
				new_file = False
			elif len(line) == 1:
				data[new_filename] = new_list
				new_filename = ""
				new_list = []
				new_file = True
			else:
				candidate = line.strip()
				new_tup = (candidate, "no")
				new_list.append(new_tup)
	if new_filename != "":
		data[new_filename] = new_list
	return data

def calc_train_features(train_data):
	train_featureset = []
	for fname, phrases in train_data.items():
		file_id_index = fname.rfind("/")
		file_id = (fname[file_id_index+1:])[0:-4]
		tfidf_list, word_list = get_tfidf_vector("TFIDF_VECTORS/" + file_id + "-gold.tfvec")
		phrases = [x for x in phrases if x[0] in word_list]
		c = FeatureCalculator(fname, phrases, tfidf_list)
		for s in c.get_phrases():
			train_featureset.append((c.get_features(s[0]), s[1]))

	return train_featureset

def get_tfidf_vector(tfidf_file_name):
	tfidf_vec = []
	word_list = []
	#check_vec = []
	with open(tfidf_file_name, "r") as f:
		for line in f:
			words = line.strip().split(':')
			tfidf_vec.append(float(words[1]))
			word_list.append(words[0])
			#check_vec.append(words[0] + ": " + words[1])
	return tfidf_vec, word_list #check_vec


def run_classifier(classifier, featureset):
	file = open(outputPath, "w") #MAXENT CHANGE
	cur_filename = ""
	for s in featureset:
		feat = s[2]
		result = classifier.prob_classify(feat[0])
		if (result.prob("yes") > .9):
			#print formatted result
			if cur_filename == "":
				cur_filename = s[0]
				file.write(cur_filename + "\n")
				#print cur_filename
			elif cur_filename != s[0]:
				cur_filename = s[0]
				file.write("\n")
				#print '\n'
				#print cur_filename
				file.write(cur_filename + "\n")
			file.write(s[1] + "\n")
	file.close()

def calc_test_features(test_data):
	test_featureset = []
	keylist = test_data.keys()
	keylist = sorted(keylist)
	for fname in keylist:
		file_id_index = fname.rfind("/")
		file_id = (fname[file_id_index+1:])[0:-4]

		tfidf_list, word_list = get_tfidf_vector("TFIDF_VECTORS/600test/" + file_id + ".tfvec")
		phrases = test_data[fname]
		phrases = [x for x in phrases if x[0] in word_list]
		c = FeatureCalculator(fname, phrases, tfidf_list) 
		for s in c.get_phrases():
			test_featureset.append((fname, s[0], (c.get_features(s[0]), s[1])))
	return test_featureset


def processArgs():#MAXENT CHANGE
	global outputPath, candidatesPath
	myArgs = sys.argv;
	if len(myArgs) > 2:
		outputPath = myArgs[1]
		candidatesPath = myArgs[2]
	print("outputPath: " + outputPath)
	print("candidatesPath: " + candidatesPath)

def main():
	processArgs()
	init_global_vars()
	train_data = parse_train_data("train.info") #source of feature info
	test_data = parse_test_data(candidatesPath)
	train_featureset = calc_train_features(train_data)
	test_featureset = calc_test_features(test_data)
	print test_featureset
	
	#MAXENT CHANGE
	encoding = maxent.TypedMaxentFeatureEncoding.train(train_featureset, count_cutoff=3, alwayson_features=True)#MAXENT CHANGE
	classifier = maxent.MaxentClassifier.train(train_featureset, algorithm='iis', bernoulli=False, encoding=encoding, trace=3)#MAXENT CHANGE
	
	run_classifier(classifier, test_featureset)


if __name__ == "__main__":
	main()
