#!/usr/bin/env python

import sys

def read_mapper_input(stdin):
    """
    Generator to limit memory usage while reading input.
    """
    for line in stdin:
        yield line.rstrip()

def combinations(iterable, r):
    """
    Implementation of itertools combinations method. Re-implemented here because
    of import issues in Amazon Elastic MapReduce. Was just easier to do this than
    bootstrap. More info here: http://docs.python.org/library/itertools.html#itertools.combinations
    
    Input/Output:
    
    combinations('ABCD', 2) --> AB AC AD BC BD CD
    combinations(range(4), 3) --> 012 013 023 123
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def main():
    """
    Accepts an inverted index as input from stdin, as described below, and groups
    TF-IDF weights for each document so they can later be summed by the reducer.
    
    word        {"docid": 10.0}
    word2       {"docid1": 5.0, "docid2": weight2, etc.}
    
    Output:
    
    "docid1|docid2"        10.0
    "docid1|docid2"        5.0
    ...
    "docidx|docidy"        weight_product
    """
    input = read_mapper_input(sys.stdin)
    for line in input:
        # Eval the dict part of the inverted index.
        worddict = eval(line.split('\t')[1])
        
        # Iterate over permutations of document pairs for a given word
        for c in combinations(worddict.keys(), 2):
            # Calculate the sum of weights for a given word and document pair
            docid1, docid2 = c[0], c[1]
            number = float(worddict[docid1]) + float(worddict[docid2])
                
            # Return output in the form of a document pair and weight for
            # a given word. These will later be combined in the reducer.
            print '%s|%s\t%s' % (c[0], c[1], number)

if __name__ == "__main__":
    main()