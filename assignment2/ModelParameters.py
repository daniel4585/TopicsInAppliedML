import numpy as np
from collections import Counter
class ModelParameters(object):
    def __init__(self, hyperParams ):
        self.hyperParams = hyperParams


    def Init(self, train):

        self.vocabulary = Counter(sum(train, []))
        self.u = np.random.normal(scale=0.01, loc=0.0, size=(len(self.vocabulary), self.hyperParams.D))
        for i in range(len(self.vocabulary)):
            self.u[i] = self.u[i] / np.linalg.norm(self.u[i])
        denominator = sum([pow(x, self.hyperParams.alpha) for x in self.vocabulary.values()])

        unigramDistVec = [pow(x, self.hyperParams.alpha) // denominator for x in self.vocabulary.values()]
        # self.v = np.random.normal(scale=1. / self.K, size=(self.num_movies, self.K))
    # def sample_worf(self):


