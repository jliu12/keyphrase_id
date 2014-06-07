import glob
import sys

sourcePrefix = "results/vScoreTest/train/" #"candidates/v0/trainNGramCandidates/"
sourceSuffix = ".cand" #".cand"
sourcePath = sourcePrefix + "*" + sourceSuffix

corpusPrefix = "corpus/krapavin2000-train/"
corpusSuffix = ".txt"
targetPath = "v600CandsTrain.txt"



def processArgs():
	global corpusPrefix
	global sourcePrefix
	global sourceSuffix
	global targetPath
	myArgs = sys.argv;
	if(len(myArgs) > 3):
		corpusPrefix = myArgs[1]
		sourcePrefix = myArgs[2]
		targetPath = myArgs[3]
	else:
		print("Usage: uncombine_cand_files.py corpusPath sourcePrefix targetPath")
		exit(1)
	global sourcePath
	sourcePath = sourcePrefix + "*" + sourceSuffix
	print("sourcePath: " + sourcePath)
	print("corpusPrefix: " + corpusPrefix)
	
	

def main():
	processArgs();
	fh = open(targetPath, 'a')
	sources = glob.glob(sourcePath)
	print(str(len(sources)) + " files found")
	for filename in sources:
		docId = filename[len(sourcePrefix): -len(sourceSuffix)]
		#print(docId)
		fh.write(corpusPrefix + docId + corpusSuffix + "\n")
		with open(filename, 'r') as f:
			keyPhrases = f.readlines();
			
		#print(str(len(keyPhrases)))
		for phrase in keyPhrases:
			#print(phrase)
			fh.write(phrase) #.rstrip("\n") + )
		fh.write("\n")
	fh.close()

  
if __name__ == '__main__':
  main()