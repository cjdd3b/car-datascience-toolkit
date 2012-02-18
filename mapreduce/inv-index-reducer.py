#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys, math

# The number of doucments in your entire document set
DOCUMENT_SET_LENGTH = 100

def get_idf(docs_containing_term):
    return math.log(float(DOCUMENT_SET_LENGTH) / float(docs_containing_term))

def read_mapper_output(file, separator):
    """
    Generator to limit memory usage while reading input.
    """
    for line in file:
        yield line.rstrip().split(separator, 2)

def main(separator='\t'):
    """
    This reducer consolidates input from mapper into an inverted index.
    
    Input:

    this     1       0.5
    document 1       0.5
    document 2       1
    ...
    word    docidx   tf
    
    Output:
    
    this        {"1": 0.5}
    document    {"1": 0.5, "2", 1}
    ...
    term        {"docidx": tf, "docidy": count1 ...}
    """
    data = read_mapper_output(sys.stdin, separator)
    
    # Input from the mapper is sorted by key by map/reduce. This groups the
    # input by key and then consolidates the values.
    for current_word, group in groupby(data, itemgetter(0)):
        fileDict = {}
        # Groups mapper input by term and creates a dictionary containing the ID
        # for each document in which that term appeared, along with the frequency
        # score calculated by the mapper.
        for current_word, fileName, count in group:
            fileDict[fileName] = float(count)
        
        # Loops through each dictionary and calculates TF-IDF weights
        for item in fileDict.items():
            tf = item[1] # Term frequency from the mapper
            docs_containing_term = float(len(fileDict)) # The number of documents containing the term
            idf = get_idf(docs_containing_term) # IDF score from function above
            
            # Assign TF-IDF score to each item in the index
            fileDict[item[0]] = tf * idf
        
        # Return inverted index with TF-IDF weights
        print "%s\t%s" % (current_word, str(fileDict))

if __name__ == "__main__":
    main()