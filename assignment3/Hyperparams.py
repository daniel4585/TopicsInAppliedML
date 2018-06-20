import numpy as np
class Hyperparams(object):
    def __init__(self, maxDepth, eta=0.75, nu=1.0, numThresholds=np.inf, numOfTrees=100, minNodeSize=10):
        self.minNodeSize = minNodeSize
        self.numOfTrees = numOfTrees
        self.maxDepth = maxDepth
        self.eta = eta
        self.nu = nu
        self.numThresholds = numThresholds



    def __str__(self):
        return "Max Depth: " + str(self.maxDepth) + " Eta: " + str(self.eta) + " Nu: " + str(
            self.nu) + " Num Thresholds: " + str(self.numThresholds) + " Num Of Trees: " + str(
            self.numOfTrees) + " Min Node Size: " + str(self.minNodeSize)
