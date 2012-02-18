"""
similarity.py

Below are implementations of five common similarity metrics used in data and text
mining applications. Each of the metrics takes two vectors as input, represented
by Python lists.

Most of the similarity metrics in this package are most easily conceptualized in
geometric spaces. As such, each vector in this context typically represents a point
in n-dimensional space. For example:

>> print euclidean([1,2], [2,1])

More information on each of these metrics can be found here:

Euclidean distance: http://en.wikipedia.org/wiki/Euclidean_distance
Jaccard similarity: http://en.wikipedia.org/wiki/Jaccard_index
Hamming distance: http://en.wikipedia.org/wiki/Hamming_distance
Pearson correlation: http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
Cosine distance: http://en.wikipedia.org/wiki/Cosine_similarity
"""
import math
import operator
from itertools import imap

def euclidean(v1, v2):
    """
    Think of Euclidean distance as "as the crow flies" distance between two
    n-dimensional vectors. It is often used in things like clustering algorithms
    to determine whether points are close enough together to fit into a cluster,
    but it can also be used to determine similarity for tasks like recommendation.
    
    One warning: The more dimensions your vector has, the less useful Euclidean
    distance is. This is known as the "Curse of Dimensionality." More on that here:
    http://en.wikipedia.org/wiki/Curse_of_dimensionality
    """
    try: # Test to ensure vectors are the same length
        assert len(v1) == len(v2)
    except AssertionError, e:
        raise(AssertionError("Vectors must be same length!"))
        
    sum = 0 # Compute sum of squares
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return 1 / (1 + (math.sqrt(sum))) # Normalize as a score between 0 and 1

def jaccard(v1, v2):
    """
    Jaccard takes two vectors and computes a score based on the number of items
    that overlap. Specifically, it is defined as the number of items contained
    in both sets (the intersection) divided by the total number of items in both
    sets combined (the union).
    
    This metric can be useful for calculating things like string similarity. A
    variation on this metric, described on its Wikipedia page, is especially helpful
    for measuring binary "market basket" similarity.
    """
    intersection = list(set(v1) & set(v2))
    union = list(set(v1) | set(v2))
    # Subtracting from 1.0 converts the measure into a distance
    return 1.0 - float(len(intersection)) / float(len(union))

def hamming(v1, v2):
    """
    Hamming distance is a measure of similarity that takes into account the order
    of items in a sequence. Looked at another way, it represents the number of
    changes that would need to be made for two sequences (or strings) to be made
    identical.
    
    It can be useful for applications such as comparing categorical data over a time
    series. It is especially useful for binary data and can be used for tasks like
    anomaly detection.
    """
    try: # Test to ensure vectors are the same length
        assert len(v1) == len(v2)
    except AssertionError, e:
        raise(AssertionError("Vectors must be same length!"))
    # Iterate over each vector and test every item against each other in sequence
    return sum(i != j for i, j in zip(v1, v2))
    
def pearson(v1, v2):
    """
    Pearson distance measures the degree to which two vectors are linearly related.
    Journalists might also know it as simple correlation. We use it to determine
    whether, for example, low test scores are related to income; or large sums
    of campaign contribution are related to years of incumbency.
    
    In data mining, Pearson can be useful in determining similarity for the
    purposes of recommendation. It can also be more useful than Euclidean distance
    in cases where data is not well normalized. The value will always be between
    -1 and 1, with 0 indicating no correlation, -1 indicating a perfect negative
    correlation and 1 indicating a perfect positive correlation.

    This particularly concise implementation comes from Cloudera chief scientist Jeff
    Hammerbacher: http://bit.ly/wNIgqu
    """
    try: # Test to ensure vectors are the same length
        assert len(v1) == len(v2)
    except AssertionError, e:
        raise(AssertionError( "Vectors must be same length!"))
    n = len(v1) # Length of both vectors (because they have to be the same)
    
    sum_x = float(sum(v1)) # Sum of all items in vector v1
    sum_y = float(sum(v2)) # Sum of all items in vector v2
    sum_x_sq = sum(map(lambda x: pow(x, 2), v1)) # Sum of squares in v1
    sum_y_sq = sum(map(lambda x: pow(x, 2), v2)) # Sum of squares in v2
    psum = sum(imap(lambda x, y: x * y, v1, v2))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

def cosine(v1, v2):
    """
    Cosine distance measures the similarity of two vectors by taking the cosine
    of the angle between them. It returns a value between -1 and 1, with -1 meaning
    the two vectors are exactly the opposite and 1 meaning they are exactly the same.
    
    Cosine distance is commonly used in text mining to compare document similiarities,
    typically by being applied to TF-IDF vector outputs. Like the other metrics
    in this library, it has many other applications as well.
    
    Note that cosine distance doesn't take magnitude into account, meaning it doesn't
    pay attention to the number of times a given word is listed in a document.
    """
    try: # Test to ensure vectors are the same length
        assert len(v1) == len(v2)
    except AssertionError, e:
        raise(AssertionError("Vectors must be same length!"))
        
    n = len(v1) # Length of both vectors (because they have to be the same)
    
    # Calculate dot product of the two input vectors
    dot = sum([v1[i] * v2[i] for i in range(n)])
    # Normalize the two input vectors so a cosine can be calculated between them
    norm1 = math.sqrt(sum([v1[i] * v1[i] for i in range(n)]))
    norm2 = math.sqrt(sum([v2[i] * v2[i] for i in range(n)]))
    return dot / (norm1 * norm2) # Return the cosine of the angle

if __name__ == '__main__':
    print jaccard([4,12,31,6], [4,5,9,4])
    #print tanimoto([1,0,1,1,0], [1,1,0,0,1])