import re
import os
import sys
import os.path
import fnmatch
import glob
import collections
import json
import math

#defaults
corpusName = "corpus/dummy"
versionName = "unknown";
dataSet = "train"
predSuffix = ".pred"

pScoresTotal = 0
rScoresTotal = 0
fScoresTotal = 0
numDocs = 0

def processArgs():
	global corpusName, versionName, dataSet, predSuffix
	myArgs = sys.argv;
	if len(myArgs) > 3:
		corpusName = myArgs[1]
		versionName = myArgs[2];
		dataSet = myArgs[3];
		#predSuffix = myArgs[4]
	else:
		print("======================================")
		print("USAGE: python substring_scorer [corpusName] [predictorVersionName] [train|dev|test] [predSuffix] \n Continuing using defaults:")
	print("======================================")
	print("corpusName: " + corpusName + "\nversionName: " + versionName + "\ndataSet: " + dataSet)
	print("======================================")

def processPredictions():
	predPath = "results/" + versionName + "/" + dataSet + "/*" + predSuffix;
	print("predPath: "  + predPath)
	suffixLength = len(predSuffix)
	prefixLength = len(predPath) - len("*" + predSuffix);
	predGlob = glob.glob(predPath);
	for predFilename in predGlob:
		docId = predFilename[prefixLength: -suffixLength]
		keyFilename = corpusName + "-" + dataSet + "/" +  docId + ".key"
		#TESTING
		print(str(docId))
		#endTesting
		computeScoresForSingleDoc(set(getKeyPhrasesFromAsList(predFilename)), set(getKeyPhrasesFromAsList(keyFilename)))
		global numDocs
		numDocs += 1

def calcTPandFN(predPhrases, goldPhrases):
	tp = 0
	fn = len(goldPhrases)
	matched_set = set()
	for s in predPhrases:
		for g in goldPhrases:
			if s.find(g) != -1 or g.find(s) != -1:
				cur_length = len(matched_set)
				matched_set.add(g)
				if (len(matched_set) > cur_length):
					tp += 1
	fn = fn - len(matched_set)
	return tp, fn

# computes scores with substrings
def computeScoresForSingleDoc(predPhrases, goldPhrases):
	global pScoresTotal, rScoresTotal, fScoresTotal

	int_TP, int_FN = calcTPandFN(predPhrases, goldPhrases)#float(len(predPhrases.intersection(goldPhrases)))
	TP = float(int_TP)
	FN = float(int_FN)
	FP = float(len(predPhrases) - TP)#float(len(predPhrases.difference(goldPhrases)))
	#FN = float(len(goldPhrases.difference(predPhrases)))
	prec = TP / (TP + FP);
	rec = TP/(TP + FN);
	f1 = 0.0
	if (prec + rec > 0):
		f1 = 2.0 * prec * rec/ (prec + rec)
	pScoresTotal += prec;
	rScoresTotal += rec;
	fScoresTotal += f1
	# testing
	print("Prec: " + str(prec) + " rec: " + str(rec) + " f1: " + str(f1))
	#endTesting  


def getKeyPhrasesFromAsList(filename):
	with open(filename, 'r') as f:
		keyPhrases = f.readlines();
	return keyPhrases

def reportOverallStats(statsFileName):
	
	overallP = pScoresTotal/ float(numDocs)
	overallR = rScoresTotal/ float(numDocs)
	overallF = fScoresTotal/ float(numDocs)
	resultString = "==================\n"
	resultString += "overall scores:\n"
	resultString += ("Average Precision " + str(overallP))
	resultString += ("\nAverage Recall " + str(overallR))
	resultString += ("\nAverage F1: " + str(overallF))
	if(overallP + overallR > 0):
		resultString += ("\nF1 of Avg P and R: " + str(2.0 * overallP * overallR / (overallP + overallR)))
	print(resultString)
	with open(statsFileName + ".txt", 'w') as f:
		f.write(resultString)
    
def main():
	processArgs();
	processPredictions()
	reportOverallStats("stats/" + versionName + "-" + dataSet)
  
if __name__ == '__main__':
  main()
