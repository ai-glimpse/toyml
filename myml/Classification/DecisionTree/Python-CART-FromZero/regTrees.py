import numpy as np
from numpy import shape, power

def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            # Map everything to float
            fltLine = [float(i) for i in curLine]
            dataMat.append(fltLine)
    return np.matrix(dataMat)

def binSplitDataSet(dataSet, feature, value):
    """
    >>> testMat = np.mat(np.eye(4))
    >>> mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    >>> mat0
    matrix([[0., 1., 0., 0.]])
    >>> mat1
    matrix([[1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """
    ##############################################
    #Here we drop '[0]' in the end of source code#
    ##############################################
    mat0 = dataSet[np.nonzero(dataSet[:, feature]  > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    # get total square error from variance(MSE)
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType, errType, ops):
    # A totlerance on the error reduction
    tolS = ops[0]
    # The minimum data instances to include in a split
    tolN = ops[1]
    # Eixit 1: all cases are of same class
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    # Initialization for search
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featureIndex in range(n-1):
        # we use 'np.unique' rather than 'set(...)' 
        for splitValue in np.unique(np.array(dataSet[:, featureIndex])):
            mat0, mat1 = binSplitDataSet(dataSet, featureIndex, splitValue)
            # if over-split then pass
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # Update the state in search procedure
            if newS < bestS:
                bestIndex = featureIndex
                bestValue = splitValue
                bestS = newS
    # Exit 2: low error reduction(prepruning)
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # Exit 3: split creates small dataset
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    #>>> myDat = loadDataSet('ex0.txt')
    #>>> myMat = np.mat(myDat)
    #>>> createTree(myMat)
    """
    feature, value = chooseBestSplit(dataSet, leafType, errType, ops)
    if feature == None:
        return value
    retTree = {}
    retTree['splitIndex'] = feature
    retTree['splitValue'] = value
    lSet, rSet = binSplitDataSet(dataSet, feature, value)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# Regression tree-pruning functions
def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['splitIndex'], tree['splitValue'])
        if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['splitIndex'], tree['splitValue'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree

if __name__ == '__main__':
    myDat = loadDataSet('ex2.txt')
    myMat = np.mat(myDat)
    tree = createTree(myMat, ops=(0, 1))
    # print(tree)
    myDatTest = loadDataSet('ex2test.txt')
    myMatTest = np.mat(myDatTest)
    tree_pruned = prune(tree, myMatTest)
    print(tree_pruned)