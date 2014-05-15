import re
import os
import sys
import os.path
import fnmatch
import glob
import collections


path = "dummy/*.txt"

def parse():
	term_counter = collections.Counter()
	for filename in glob.glob(path):
		words = re.findall('\w+', open(filename).read().lower())

		# From http://stackoverflow.com/questions/9082099/python-loop-through-files-in-a-folder-and-do-a-word-count
		ignore = ['the','a','if','in','it','of','or','on','and','to', 'is', 'for', 'that']
		counter = collections.Counter(x for x in words if x not in ignore)
		term_counter += counter

	print term_counter


def main():
	#path = input("Enter file and path, place ' before and after the path: ")
	parse()

if __name__ == "__main__":
    main()