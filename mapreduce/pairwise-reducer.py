#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys

# Only print output that has a score exceeding this number. Defining this number
# is a little tricky in this implementation, given that the output scores aren't
# normalized. You'll want to play with this until you find a threshhold that matches
# your application. Too high and some real matches might be omitted. Too low and
# you're likely to have millions, if not hundreds of millions, of lines of output.
OUTPUT_THRESHOLD = 1.0

def read_mapper_output(file, separator):
    """
    Generator that yields lines from the mapper.
    """
    for line in file:
        yield line.rstrip().split(separator, 2)

def main(separator='\t'):
    """
    Consolidates output from the mapper and sums document pair similarity scores to
    calculate a final similarity score for each document pair. Input comes as key/
    value with key being a document pair and value being the product of term weights
    therein.
    
    Input:
    
    "docid1|docid2"        10.0
    "docid1|docid2"        5.0
    ...
    "docidx|docidy"        weight_sum
    
    Output:
    
    "docid1|docid2"        15.0
    ...
    "docidx|docidy"        weight_sum
    """
    data = read_mapper_output(sys.stdin, separator)
    
    # Input from the mapper is sorted by key by map/reduce. This groups the
    # input by key and then consolidates the values.
    for docset, group in groupby(data, itemgetter(0)):
        doc1, doc2 = docset.split('|')
        totcount = 0.0
        for docset_inner, count in group:
            totcount += float(count)
        
        # Get document IDs for comparison purposes below.
        docid1 = doc1[:8]
        docid2 = doc2[:8]
        
        # Only print the output if the output score reaches a certain threshold,
        # and don't produce output for a document compared against itself.
        if (not docid1 == docid2) and totcount > OUTPUT_THRESHOLD:
            print "%s\t%s" % (docset, totcount)

if __name__ == "__main__":
    main()