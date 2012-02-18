'''
knn.py

An crude implementation of a simple k Nearest Neighbors classifier.

kNN is one of the simplest classification algorithms available. It works by classifying
items base on the closest training examples in a given feature space. For example,
picture a map of California with two pins in it -- one in Los Angeles and one in
San Francisco. If someone places a third pin next to the one in Los Angeles, a kNN
classifier would associate it with L.A. as opposed to San Francisco.

The map example is a simple one, but kNN can be used to classifiy objects in n
dimensions based on a variety of distance metrics, making it a versatile and useful
tool. On the down side, it is also computationally intensive, which becomes a bigger
problem as your input dataset grows.

As with many machine learning algorithms, kNN can be implemented in a number of different
ways. This algorithm could easily be adapated to predict continuous variables (like price),
to find clusters of similar words, or to weight points based on their proximity to
an input vector.

More information about k Nearest Neighbors can be found here:

Toby Segaran's Programming Collective Intelligence:
http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325
http://blog.kiwitobes.com/?p=44

Seth Daughtery's data mining portfolio:
http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/sdaugherty/knn.html

A Programmer's Guide to Data Mining:
http://guidetodatamining.com/home/toc/chapter-5/
'''
import math
import operator
from similiarity.similarity import euclidean

class kNNClassifier(object):
    def __init__(self, data):
        self.data = data # Input data
        self.distancelist = [] # Ranked list of distances of points from an input vector

    def _getdistances(self, v1, distance=euclidean):
        """
        Method that returns a list of vectors from a dataset, ranked by their proximity
        to an input vector v1. This step passes for training in k Nearest Neighbors,
        populating a list so that the top k items can be used to calculate a result.
        """
        for i in range(len(self.data)):
            v2 = self.data[i]['vector']
            self.distancelist.append((distance(v1, v2), data[i]['class']))
        self.distancelist.sort(key=lambda x: x[0], reverse=True)
        return

    def classify(self, v1, k=3):
        """
        A simple form of a k Nearest Neighbors classifier. Finds the k points that
        are nearest to input vector v1, as calculated by the _getdistances function
        above. It then loops through those k neighbors and simply tallies up how often
        each class appears. The class that appears the most often is assigned to
        the input vector.
        
        This illustration makes the process a little more clear:
        http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
        
        Often, classifiers will also be weighted so that neighbors that are closer
        to the input vector are worth more than those farther away. This classifier
        can also be adapted to predict continuous data, rather than just categories --
        for example, the price of a rare baseball card based on its age, condition
        and the Hall of Fame status of its player. A great implementation of that
        type of classifier can be found in Toby Segaran's book, Programming Collective
        Intelligence.
        """
        # First, calculate the distances between v1 and all items in the dataset
        self._getdistances(v1)
        klasses = {}
        # For each neighbor in k, tally up the class values
        for i in range(k):
            klass = self.distancelist[i][1]
            klasses.setdefault(klass, 0)
            klasses[klass] += 1
        # Sort the classes based on their final counts
        finalcounts = sorted(klasses.iteritems(), key=operator.itemgetter(1))
        finalcounts.reverse()
        # Return the class value (or values) with the highest counts
        return [x[0] for x in finalcounts if x[1] == finalcounts[0][1]]

if __name__ == '__main__':
    data = [
        {'class': 'a', 'vector': (1, 2)},
        {'class': 'a', 'vector': (2, 1)},
        {'class': 'b', 'vector': (3, 5)},
        {'class': 'c', 'vector': (4, 2)},
    ]
    c = kNNClassifier(data)
    print c.classify((1,3))