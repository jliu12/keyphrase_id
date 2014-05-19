import nltk
from nltk.classify import apply_features

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
		print new_dict
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

	def ft_tfidf(self, candidate):
		return{'tfidf': self.tfidf[candidate]}


	def get_phrases(self):
		return self.phrases

	#calculates a dictionary of all the features for a word
	def get_features(self, candidate):
		feature_dict = self.ft_keyphrase_len(candidate)
		feature_dict.update(self.ft_first_occurrence_position(candidate))
		feature_dict.update(self.ft_tfidf(candidate))
		return feature_dict

def main():
	tfidf_list = [0.9, 0.8, 0.6, 0.4, 0.5, 0.4, 0.1, 0.2, 0.3, 0.2, 0.1]
	phrases = [("petri nets", "yes"), ("reachability analysis", "yes"), ("ada tasking", "yes"), ("deadlock analysis", "yes"), ("net reduction", "yes"), ("concurrent software", "yes"),
					("as part", "no"), ("major difficulty", "no"), ("with regards", "no"), ("analysis problems", "no"), ("previously defined", "no")]
	c = FeatureCalculator("corpus/dummy/245603.txt", phrases, tfidf_list)
	featureset = []
	for s in c.get_phrases():
		featureset.append((c.get_features(s[0]), s[1]))
	#featureset = []
	#featureset += apply_features(c.ft_keyphrase_len, c.get_phrases())
	#featureset += apply_features(c.ft_keyphrase_len, c.get_phrases())
	print(featureset)
	classifier = nltk.NaiveBayesClassifier.train(featureset)
	print (classifier.classify(c.ft_keyphrase_len("are not")))

if __name__ == "__main__":
	main()