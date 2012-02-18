#!/usr/bin/python

import sys

def freq(word, document):
  """
  Calculates the number of times a word appears in a document.
  """
  return document.split(None).count(word)
  
def wordCount(document):
  """
  Calculates the word count of a document.
  """
  return float(len(document.split(None)))

def tf(word, document):
  """
  Calculates the frequency of a word in a document as a percentage of the
  total words in that document. This is the first step in TF-IDF weighting.
  
  For example, if a document is 100 words long, and the word cat appears three
  times, the term frequency of cat will be 0.03.
  """
  return freq(word,document) / wordCount(document)

def read_mapper_input(stdin):
    """
    Generator to limit memory usage while reading input. Using generators rather
    than lists (which load data into memory) are a good way to save resources to
    keep your MapReduce jobs running quickly.
    """
    for line in stdin:
        yield line.rstrip()

def main():
    """
    The first step in this comparison process is to create an inverted index to
    make document comparison faster and more efficient.
    
    More info on that here: http://en.wikipedia.org/wiki/Inverted_index
    
    This mapper takes input as a document ID and string of words and returns a
    list of term weights, which will be consolidated into a true inverted index
    in the reducer.
    
    Input:
    
    docid|This is some document text
    docid2|And this is another document
    
    Output:
    
    this     1       0.5
    document 1       0.5
    document 2       1
    ...
    word    docidx   tf 
    """
    for line in read_mapper_input(sys.stdin):
        # Split document ID and document string
        docid = line.split('|')[0]
        document = line.split('|')[1]
        
        frequencies = {}
        # Crudely tokenize document into words and tally up word counts. This
        # works best if preprocessing strips punctuation, removes stopwords,
        # performs stemming, etc.
        for word in document.split():
            try:
                frequencies[word] += 1
            except KeyError:
                frequencies[word] = 1
        
        # Print term frequencies to stdout for ingestion by reducer.
        for word in frequencies:
            print '%s\t%s\t%s' % (word, docid, tf(word, document))

if __name__ == "__main__":
    main()