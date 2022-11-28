from toyml.utils.types import Vector, DataSet, Label, Labels


class NaiveBayesClassifier:
    """Naive Bayes Classifier implementation.

    Ref:
    1. Li Hang
    2. Alan Ritter
    3. Murphy
    """
    def __init__(self, dataset: DataSet, labels: Labels) -> None:
        self._dataset = dataset
        self._labels = labels
        # dimensions
        # N: sample size
        self._N = len(labels)
        # n: feature num
        self._n = len(self._dataset[0])
        # class num
        self._k = len(set(labels))
        # model data
        self.classProb = None
        self.condProb = None

    def fit(self):
        # calc class prob: P(Y=Ck)
        self.classProb = {}
        for y in self._labels:
            self.classProb[y] = self.classProb.get(y, 0) + 1 / self._N

        # calc conditional prob: P(Xj=ajl | Y=Ck)
        # we use a matrix of dim (k, n) with element {aj1:xxx, ..., ajSj:xxx}
        # to save the result
        self.condProb = [[{} for col in range(self._n)]
                         for row in range(self._k)]
        # calc with "sorted" samples
        indexes = sorted(range(self._N), key=lambda i: self._labels[i])
