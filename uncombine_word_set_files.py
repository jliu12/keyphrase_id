import glob
import sys


sourcePath = "naive_bayes_results.txt"
corpusPrefix = "corpus/krapavin2000-test/"
corpusSuffix = ".txt"
targetPath = "results/v0/scoreTestSimple/"
targetSuffix = ".pred"


def processArgs():
	global targetPath
	global targetSuffix
	global sourcePath
	myArgs = sys.argv;
	if(len(myArgs) > 2):
		targetPath = myArgs[1]
		targetSuffix = myArgs[2]
		sourcePath = myArgs[3]
	print("sourcePath: " + sourcePath)
	print("targetPath: " + targetPath + " * "  + targetSuffix)
	
	

def main():
	processArgs();
	new_file = True
	docId = ""
	new_list = []
	with open(sourcePath, "r") as f:
		for line in f:
			if new_file:
				new_file = False;
				new_filename = line.strip()
				docId = new_filename[len(corpusPrefix): -len(corpusSuffix)]
			elif len(line) == 1:
				#write to file:
				new_path = targetPath + docId + targetSuffix
				with open(new_path, 'w') as nf:
					for phrase in new_list:
						nf.write(phrase + "\n")
				#reset vars
				new_file = True
				new_list = []
				docId = ""
			else:
				new_list.append(line.strip())

  
if __name__ == '__main__':
  main()