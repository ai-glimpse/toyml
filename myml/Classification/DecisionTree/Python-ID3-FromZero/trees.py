import copy
import functools
from math import log
from collections import Counter

# Create a toy data set
def createDataSet():
    dataset = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
            ]
    # feature names
    labels = ['no surfacing', 'flippers']
    return dataset, labels

# Function to calculate the Shannon entropy of a dataset
def calcShannonEnt(dataSet):
    """
    >>> dataSet, labels = createDataSet()
    >>> entropy = calcShannonEnt(dataSet)
    >>> abs(entropy-(-0.4*log(0.4, 2)-0.6*log(0.6, 2))) < 0.001
    True
    """
    n = len(dataSet)
    if n == 0 or n == 1:
        return 0
    classes = [case[-1] for case in dataSet]
    count_obj = Counter(classes)
    return sum([-(Ck/n)*log(Ck/n, 2) for Ck in count_obj.values()])

# Split dataset with specifice feature(axis is it's number) and value.
# So the operation can be used only for category features.
# Note that the feature used to split the dataset will be removed in subest.
def splitDataSet(dataSet, axis, value):
    """
    >>> myDat, labels = createDataSet()
    >>> splitDataSet(myDat, 0, 1)
    [[1, 'yes'], [1, 'yes'], [0, 'no']]
    >>> splitDataSet(myDat, 0, 0)
    [[1, 'no'], [1, 'no']]
    """
    # extend will return None, so we use '+' here
    return [case[:axis] + case[axis+1:]
            for case in dataSet if (case[axis] == value)]


# Two questions that must be considered:
# Q1: Which feature?
# Q2: Which value?
# Return best feature' index
def chooseBestFeatureToSplit(dataSet):
    """
    >>> myDat, labels = createDataSet()
    >>> chooseBestFeatureToSplit(myDat)
    0
    """
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    # Which feature
    for axis in range(numFeatures):
        newEntropy = 0.0
        # Which value?
        uniqueVals = set([case[axis] for case in dataSet])
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, axis, value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = axis
    return bestFeature

# Get major class from class list(with labels)
def majorityCnt(classList):
    return Counter(classList).most_common(1)[0][0]

################################################################
# Here we defined a decorator for pass labels(a list) as value.#
# If not, labels will be changed after we build the tree.      #
# ##############################################################
# Tools function
def pass_by_value(f):
    @functools.wraps(f)
    def _f(*args, **kwargs):
        args_copied = copy.deepcopy(args)
        kwargs_copied = copy.deepcopy(kwargs)
        return f(*args_copied, **kwargs_copied)
    return _f

# Tree-building code
@pass_by_value
def createTree(dataSet, labels):
    """
    >>> myDat, labels = createDataSet()
    >>> createTree(myDat, labels)
    {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    """
    classList = [case[-1] for case in dataSet]
    # Stop cond 1: all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # Stop cond 2: no more features, return mojority
    # In this case, dataSet are just class list
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # How to build a tree
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLable = labels[bestFeature]
    myTree = {bestFeatureLable:{}}
    # get sub-labels
    del(labels[bestFeature])
    # get unique feature values of best feature
    uniqueVals = set([case[bestFeature] for case in dataSet])
    for value in uniqueVals:
        # make a copy
        subLabels = labels[:]
        myTree[bestFeatureLable][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree



# Classification function for an existing decision tree
def classify(inputTree, featureLabels, testVec):
    """
    >>> myDat, labels = createDataSet()
    >>> tree = createTree(myDat, labels)
    >>> classify(tree, labels, [1, 0])
    'no'
    >>> classify(tree, labels, [1, 1])
    'yes'
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # Translate label string to index
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key], featureLabels, testVec)
            else:
                return secondDict[key]

# Methods for persisting the tree with pickle
# #############################################################
# Here, in Python3, we should use 'wb' rather than 'w'        #
# ref:https://github.com/smallcorgi/Faster-RCNN_TF/issues/170 #
###############################################################
def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)

def grabTree(filename):
    """
    >>> myDat, labels = createDataSet()
    >>> tree = createTree(myDat, labels)
    >>> storeTree(tree, 'classiffierStorage.txt')
    >>> grabTree('classiffierStorage.txt')
    {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Example
if __name__== '__main__':
    with open('lenses.txt') as fr:
        lenses = [case.strip().split('\t') for case in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # Plot the tree
    from treePlotter import createPlot
    createPlot(lensesTree)