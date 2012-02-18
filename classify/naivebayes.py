'''
naivebayes.py

An implementation of a simple Naive Bayes classifier, adapted from the excellent
Programmer's Guide to Data Mining, which can be found here:

http://guidetodatamining.com/home/toc/chapter-5/

Naive Bayes is considered one of the simplest but most effective classifiers in
machine learning. It is considered a supervised classification method because it
relies on a set of training data in which items and their proper classifications
are already known. The algorithm uses probabilities to "learn" about the characteristics
of those items and then uses that knowledge to classify new, unknown observations.

For example, the algorithm might learn that an apple is a red fruit with a stem
and an orange is an orange fruit with a rind. If the classifier then receives an
input of a green fruit with a stem, it will classify the item as an apple.

This implementation is set up to deal only with discrete categorical data, which
is typical behavior for a simple Naive Bayes classifier. However, Naive Bayes can
deal with continuous numerical data as well using one of two approaches: making
the continuous data discrete by fitting it into bins, or by assuming the data fits
a Gaussian (normal) distribution and altering the algorithm accordingly.

A good explanation of these approaches can be found here under the "Parameter
Estimation" section: http://en.wikipedia.org/wiki/Naive_Bayes_classifier

Other useful information about Naive Bayes can be found here:

Naive Bayes for text classification in NLTK:
http://nltk.googlecode.com/svn/trunk/doc/book/ch06.html
https://gist.github.com/1266498

Toby Segaran's Programming Collective Intelligence:
http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325
http://blog.kiwitobes.com/?p=44
'''

class NaiveBayes(object): 
    def __init__(self, data):
        """
        Bayesean classifiers require two types of probabilities to be created in
        training in order to properly classify input. They are known as "prior"
        and "conditional" probabilities.

        Priors are sort of a starting point for Bayesean probability, based on
        the classes but not their attributes. For example, if there are 10 items
        in the dataset -- 8 of type 'A' and 2 of type 'B' -- the model understands
        that a new item is far more likely to be of type 'A' (.8 probability)
        than type 'B' (.2 probability), even before its individual attributes
        are taken into account.

        The classifier also needs access to a set of conditional probabilities
        based on the features of each class. For instance, the probability that
        a certain attribute is associated with type 'A' vs. type 'B'.
        """
        self.data = data # Input. Assumes first item in each row is class name.    
        self.prior = {}
        self.conditional = {}

    def _calculate_prior(self, total, classes):
        """
        Calculate prior probabilities, as described above, using the training set.
        
        This particular approach calculates class probabilities based on the frequency
        of each class in the training set. Another approach would be to assume all
        classes are equally likely to occur.
        """
        for (category, count) in classes.items():
            self.prior[category] = float(count) / float(total)
        return

    def _calculate_conditional(self, counts, classes):
        """
        Calculate conditional probabilities, as described above, using the training
        set.
        """
        # For each class and its set of feature counts, which are represented in
        # a dictionary as numbered columns.
        for (category, columns) in counts.items():
            tmp = {}
            # For each set of feature counts in a column
            for (col, valueCounts) in columns.items():
                tmp2 = {}
                # For each feature and count in a row
                for (value, count) in valueCounts.items():
                    # Calculate the probability that each feature is associated
                    # with a given class
                    tmp2[value] = float(count) / float(classes[category])
                tmp[col] = tmp2

            tmp3 = []
            # 
            for i in range(1, len(self.data[0])):
                tmp3.append(tmp[i])
            self.conditional[category] = tmp3
        return
        
    def train(self):
        """
        Train the classifier by calculating prior and conditional probabilities
        from data provided in a training set. The larger and more varied the
        training set, the better luck you will have classifying new observations.
        """
        total = len(self.data) # Total number of items in the dataset
        classes = {} # Each distinct class in the data, with counts
        counts = {} # Counts of features, grouped under each class
        
        # For each row of data in the training set
        for instance in self.data:
            category = instance[0]
            classes.setdefault(category, 0)
            counts.setdefault(category, {})
            classes[category] += 1
            
            col = 0
            # For each column in the data row, total the rote counts of each
            # feature, grouped by the class. (Start with index 1 in the list because
            # index 0 is the category.)
            for columnValue in instance[1:]:
                col += 1
                tmp = {}
                if col in counts[category]:
                    tmp = counts[category][col]
                if columnValue in tmp:
                    tmp[columnValue] += 1
                else:
                    tmp[columnValue] = 1
                counts[category][col] = tmp
        
        # Feed those counts to the probability functions above in order to calculate
        # prior and conditional probabilities.
        self._calculate_prior(total, classes)
        self._calculate_conditional(counts, classes)
        return
    
    def classify(self, instance):
        """
        Classifies a new observation based on the probabilities calculated in
        training.
        """
        categories = {}
        # Loop through every set of conditional probabilities in the training set
        for (category, vector) in self.conditional.items():
            prob = 1
            # For every feature in each class of conditional probabilities
            for i in range(len(vector)):
                # No probability should ever be set to exactly zero, as it will
                # wipe out all other probabilities when they are multiplied.
                colProbability = .0000001
                # If a feature from the input class matches one in the training vector
                if instance[i] in vector[i]:
                    # Get the probability for that feature
                    colProbability = vector[i][instance[i]]
                # Apply each conditional probability to the total probability
                prob = prob * colProbability
            # Now apply the prior probability
            prob = prob * self.prior[category]
            categories[category] = prob
        # Total and sort all the probabilities that the classifier input 
        cat = list(categories.items())
        cat.sort(key=lambda catTuple: catTuple[1], reverse = True)
        # Return the class with the highest probability
        return(cat[0])
        
if __name__ == '__main__':
    data  = [['i100', 'both', 'sedentary', 'moderate', 'yes'],
         ['i100', 'both', 'sedentary', 'moderate', 'no'],
         ['i500', 'health', 'sedentary', 'moderate', 'yes'],
         ['i500', 'appearance', 'active', 'moderate', 'yes'],
         ['i500', 'appearance', 'moderate', 'aggressive', 'yes'],
         ['i100', 'appearance', 'moderate', 'aggressive', 'no'],
         ['i500', 'health', 'moderate', 'aggressive', 'no'],
         ['i100', 'both', 'active', 'moderate', 'yes'],
         ['i500', 'both', 'moderate', 'aggressive', 'yes'],
         ['i500', 'appearance', 'active', 'aggressive', 'yes'],
         ['i500', 'both', 'active', 'aggressive', 'no'],
         ['i500', 'health', 'active', 'moderate', 'no'],
         ['i500', 'health', 'sedentary', 'aggressive', 'yes'],
         ['i100', 'appearance', 'active', 'moderate', 'no'],
         ['i100', 'health', 'sedentary', 'moderate', 'no']]
    
    b = NaiveBayes(data)
    b.train()
    print b.classify(['health', 'moderate', 'moderate', 'yes'])
    print b.classify(['appearance', 'moderate', 'moderate', 'no'])