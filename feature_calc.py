import nltk
import scipy
from nltk.classify import apply_features
from nltk.corpus import stopwords

#global variables for classification
stopwords = []

class FeatureCalculator:
	def __init__(self, document, phrases, tfidf):
		self.phrases = phrases
		#self.tfidf = self.read_tfidf(tfidf) # this is a dict!!!
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

	#TODO: implement
	def ft_doc_section(self, candidate):
		#not yet implemented, need document section markers
		return {'section': 1}

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

		#tfidf = self.ft_tfidf(candidate)
		#feature_dict.update(tfidf)

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
	tfidf_list = [0.9, 0.8, 0.6, 0.4, 0.5, 0.4, 0.1, 0.2, 0.3, 0.2, 0.1]
	for fname, phrases in train_data.items():
		c = FeatureCalculator(fname, phrases, tfidf_list)
		for s in c.get_phrases():
			train_featureset.append((c.get_features(s[0]), s[1]))
	return train_featureset

def run_classifier(classifier, featureset):
	cur_filename = ""
	for s in featureset:
		result = classifier.classify(s[0])
		if (result == "yes"):
			#print formatted result
			if cur_filename == "":
				cur_filename = "the right filename"
				print cur_filename
			elif cur_filename != "the right filename"
				cur_filename = "the right filename"
				print "new line"
				print cur_filename
			print "the word"
def main():
	init_global_vars()
	train_data = parse_train_data("train.info")
	test_data = parse_test_data("testCandidatesV1.txt")
	train_featureset = calc_train_features(train_data)
	test_featureset = calc_train_features(test_data)
	#print(test_featureset)
	size = int(len(train_featureset) * 0.1)
	#train_set, test_set = train_featureset[size:], train_featureset[:size]

	classifier = nltk.NaiveBayesClassifier.train(train_featureset)
	#print nltk.classify.accuracy(classifier, test_featureset)
	run_classifier(classifier, test_featureset)


if __name__ == "__main__":
	main()