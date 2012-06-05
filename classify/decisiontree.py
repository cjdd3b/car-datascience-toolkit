'''
decisiontree.py

An implementation of a simple Naive Bayes classifier, adapted from Toby Segaran's
book Programming Collective Intelligence:

http://shop.oreilly.com/product/9780596529321.do

Decision trees are among the most intuitive and effective machine learning algorithms
available for classification tasks. In fact, variations of decision trees, such as the
Random Forest algorithm, consistently outperform more complex and sophisticated approaches.

The easiest way to understand decision trees is to think of them as a massive if-else statement.
Given a set of input data, the algorithm below will recursively generate a tree that outlines
decision paths given all possible variables in the system. The concept of a decision tree is very
clear if you look at an image of a simple one, such as this:

http://en.wikipedia.org/wiki/Decision_tree_learning

The implementation below is ridiculously oversimplified, but it should give you a sense of how
decision trees can work in practice. Common functions such as pruning, which prevents overfitting
in the dataset, have been omitted here. In practice, it's probably best to use a more robust Python
library for heavy-duty decision tree work. A couple good ones include:

PyDTL: http://scaron.info/pydtl/
scikit-learn: http://scikit-learn.org/stable/

Finally, more information about decision trees can be found here:

http://en.wikipedia.org/wiki/Decision_tree
http://en.wikipedia.org/wiki/Random_forest
'''

########## CLASSES ##########

class DecisionNode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
      self.col = col
      self.value = value
      self.results = results
      self.true_branch = tb
      self.false_branch = fb

########## FUNCTIONS ##########

def uniquecounts(rows):
    """
    Helper function to tally up the counts of unique attributes
    in a set. Used in calculating entropy.
    """
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

def entropy(rows):
    """
    Calculates the entropy of a given set.
    """
    from math import log
    log2 = lambda x:log(x) / log(2)  
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent

def divideset(rows, column, value):
    """
    Divides a set into true/false chunks needed for constructing the
    tree. Deals with both categorical and continuous data.
    """
    split_function = None
    # Turn continuous data like ints and floats into a binary
    # representation that can be split.
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row:row[column] >= value
    else:
        split_function = lambda row:row[column] == value
   
    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

def buildtree(rows):
    """
    A function for recursively building a decision tree given a set of data,
    in this case represented as a two-dimensional array of Python lists.
    """
    if len(rows)==0: return DecisionNode()
    current_score = entropy(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):

        # Loop through every column and tally up every possible
        # value for that column. Put that into a dictionary.
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        
        # Now split the data into two subsets using the divideset
        # function above.
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
      
            # Calculate information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * entropy(set1) - (1 - p) * entropy(set2)

            # Decides the best criteria on which to split the tree, such that 
            # entropy is minimized and information gain is maximized.
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain =  gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
  
    # As long as there is still information gain to be had
    if best_gain > 0:
        # Recurse and move to the next level of the tree
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        # And build out new nodes
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    
    # Once the treebuild is complete, return the full tree
    return DecisionNode(results=uniquecounts(rows))

def classify(observation, tree=None):
    """
    Function for classifying an unknown observation given a tree. In this case,
    the observation is represented as a Python list of attributes and should match
    the order and length of the items in the training set.
    """
    # Exit condition. Once results are assigned to the tree, return them and
    # stop recursion.
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        # Deal with continuous variables.
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value: branch = tree.true_branch
            else: branch = tree.false_branch
        # Deal with standard categorical data.
        else:
            # If the value matches the value of a given
            # node in the tree, move down to the next
            # branch and recurse.
            if v == tree.value: branch = tree.true_branch
            else: branch = tree.false_branch
        return classify(observation, branch)

########## MAIN ##########

if __name__ == '__main__':
    # The training data is a small slice of campaign finance data that represents
    # features in this order: amount of donation; whether it's to the opposite party
    # than a donor usually gives to; percentile rank of the donation given all donations;
    # percentile rank of the donation given donations by that specific donor; and whether
    # the contributor has given to the candidate before.

    # The final item in the row is the proper classification. In this case "interesting"
    # or "boring"

    training_data=[
        [1000000, 'y', 100.0, 100.0, 'y', 'interesting'],
        [100, 'n', 20.0, 30.0, 'n', 'boring'],
        [2000, 'y', 70.0, 60.0, 'n', 'interesting'],
        [10000, 'n', 80.0, 80.0, 'y', 'boring'],
        [500, 'y', 20.0, 10.0, 'y', 'boring'],
        [500000, 'n', 90.0, 100.0, 'y', 'interesting'],
        [15000000, 'n', 100.0, 100.0, 'y', 'interesting'],
        [13000, 'y', 70.0, 30.0, 'n', 'boring'],
        [8000, 'y', 50.0, 70.0, 'n', 'interesting'],
        [10, 'y', 0.0, 0.0, 'y', 'boring'],
        ]

    # Build the tree
    tree = buildtree(training_data)

    # Classify an unknown observation
    print classify([1000000, 'y', 100.0, 90.0, 'n', 'interesting'], tree)