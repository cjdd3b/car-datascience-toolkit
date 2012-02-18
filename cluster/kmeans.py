'''
kmeans.py

An implementation of a k-means clustering algorithm, adapted slightly from the
implementation in Toby Segaran's book Programming Collective Intelligence.

k-means is a clustering algorithm that requires only one input parameter: the
number of clusters you want it to find. The algorithm will then assign every point
in your dataset to one of those clusters. As such, this particular implementation
is best used for applications like segmenting a dataset into a groups. It is not
as useful as density-based algorithms like DBSCAN at separating clear sets of
clustered points from surrounding noise.

The algorithm works by following a four-step process:

Step 1: The algorithm randomly selects k points within the dataset (based on the number
of clusters you want it to find, also known as k). These are known as "means" of
the clusters, hence the term K Means.

Step 2: Points in the dataset are assigned to the nearest mean in order to form
k clusters.

Step 3: A new mean is calculated based on the centers of the new clusters.

Step 4: Repeat steps 2 and 3 until the means converge, yielding the final clusters.

Those steps come from this Wikipeida page, which explains the process in further
detail:

http://en.wikipedia.org/wiki/K-means_clustering

This simulator also does a great job of illustrating the process:

http://home.dei.polimi.it/matteucc/Clustering/tutorial_html/AppletKM.html
'''
import random
import math
from similiarity.similarity import euclidean

class KMeans(object):
    def __init__(self, data):
        self.rows = data
        # Get the min and max values of each dimension in the input vector
        self.ranges=[(min([row[i] for row in self.rows]), max([row[i] for row in self.rows])) 
            for i in range(len(self.rows[0]))]
        
    def cluster(self, k, distance=euclidean):
        """
        A simple implementation of a k-means clustering algorithm. The algorithm
        uses Euclidean distance by default and accepts one input paramater, k, or
        the number of clusters you want it to find.
        
        The trick is to choose a value of k that makes sense for what you're trying
        to accomplish. One rule of thumb suggests that a good starting point for k
        should be sqrt(n/2) where n is equal to the number of the points in the dataset.
        But that might not be helpful if you're trying to segment a large dataset
        into only two or three distinct clusters. Use the right value for the right
        job. More helpful info on choosing k can be found here:
        
        http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
        """
        lastmatches = None
        
        # Start by settings a random starting point for each cluster somewhere within the dataset.
        # This is like Step 1 from the Wikipedia diagram.
        clusters = [[random.random() * (self.ranges[i][1] - self.ranges[i][0]) + self.ranges[i][0] 
            for i in range(len(self.rows[0]))] for j in range(k)]
        
        # Now start iterating and adjusting the points until the clusters are accurate.
        # This implementation calls for a maximum of 100 iterations through the algorithm.
        for t in range(100):
            bestmatches = [[] for i in range(k)]
    
            # Find which centroid is the closest for each row by associating every
            # item in the dataset with the nearest mean. This is like step 2 from
            # the Wikipedia diagram.
            for j in range(len(self.rows)):
                row = self.rows[j]
                bestmatch = 0
                for i in range(k):
                    d = distance(clusters[i], row)
                    if d < distance(clusters[bestmatch], row): bestmatch = i
                bestmatches[bestmatch].append(j)

            # If the results don't change between iterations, the process is is complete
            if bestmatches == lastmatches: break
            lastmatches = bestmatches
    
            # Move the centroids to the average of their members
            for i in range(k):
                avgs = [0.0] * len(self.rows[0])
                if len(bestmatches[i]) > 0:
                    for rowid in bestmatches[i]:
                        for m in range(len(self.rows[rowid])):
                            avgs[m] += self.rows[rowid][m]
                    for j in range(len(avgs)):
                        avgs[j] /= len(bestmatches[i])
                    clusters[i] = avgs
      
        return bestmatches

if __name__ == '__main__':
    rows = [
        [1, 2],
        [3, 1],
        [4, 5],
        [20, 43],
        [30, 34],
    ]

    a = KMeans(rows)
    print a.cluster(k=2)