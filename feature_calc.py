import nltk
from nltk.classify import apply_features

class FeatureCalculator:
	def __init__(self, document):
		self.data = "Hello world"
		self.phrases = [("petri nets", "yes"), ("reachability analysis", "yes"), ("ada tasking", "yes"), ("deadlock analysis", "yes"), ("net reduction", "yes"), ("concurrent software", "yes"),
					("as part", "no"), ("major difficulty", "no"), ("with regards", "no"), ("analysis problems", "no"), ("previously defined", "no")]

		self.document = self.read_doc(document)

	def read_doc(self, f_name):
		with open (f_name, "r") as new_file:
			data = new_file.read().replace('\n', '').lower()
			return data

	#dummy function
	def print_data(self):
		print(self.data)

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
		#print(candidate)
		#print(float(self.document.find(candidate)))
		#print(len(self.document))
		return {'first_occur': relative_pos}




	def get_phrases(self):
		return self.phrases

	#calculates a dictionary of all the features for a word
	def get_features(self, candidate):
		feature_dict = self.ft_keyphrase_len(candidate)
		feature_dict.update(self.ft_first_occurrence_position(candidate))
		return feature_dict

def main():
	c = FeatureCalculator("corpus/dummy/245603.txt")
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