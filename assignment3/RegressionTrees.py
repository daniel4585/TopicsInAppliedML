from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from Queue import Queue
from copy import deepcopy as deepcopy
import sys


class RegressionTreeNode(object):
    def __init__(self, j=None, s=None, leftDescendant=None, rightDescendant=None, const=None):
        self.j = j
        self.s = s
        self.leftDescendant = leftDescendant
        self.rightDescendant = rightDescendant
        self.const = const

    def MakeTerminal(self, c):
        self.const = c

    def Split(self, j, s, cl, cr):
        self.j = j
        self.s = s
        self.leftDescendant = RegressionTreeNode(const=cl)
        self.rightDescendant = RegressionTreeNode(const=cr)


    def printSubTree(self):
        print(self.TreeToString(0))

    def TreeToString(self, tabs):
        if self.isLeaf():
            return "\t" * tabs + "return " + str(self.const) + "\n"
        strRep = "\t" * tabs + "if x['" + self.j + "'] <= " + str(self.s) + " then:\n" \
                 + self.leftDescendant.TreeToString(tabs + 1) \
                 + "\t" * tabs + "if x['" + self.j + "'] > " + str(self.s) + " then:\n" \
                 + self.rightDescendant.TreeToString(tabs + 1)
        return strRep

    def isLeaf(self):
        return self.leftDescendant is None

class RegressionTree(object):
    def __init__(self, root):
        self.root = root

    def GetRoot(self):
        return self.root

    def Evaluate(self, x):
        node = self.GetRoot()
        while True:
            if node.isLeaf():
                return node.const

            if x[node.j] <= node.s:
                node = node.leftDescendant
            else:
                node = node.rightDescendant

    def __str__(self):
        return self.GetRoot().TreeToString(0)

    def getFeatureImprortance(self, data):
        featImportance = defaultdict(int)
        self.getImportance(data, self.GetRoot(), featImportance)

        print featImportance
        return featImportance


    def getImportance(self, data, node, featImportance):
        if node.isLeaf():
            return
        pl = data.loc[data[node.j] <= node.s]
        pr = data.loc[data[node.j] > node.s]

        # calculate left node sum
        sumL = ((pl["SalePrice"] - node.leftDescendant.const)**2).sum()

        #calculate right node sum
        sumR = ((pr["SalePrice"] - node.rightDescendant.const)**2).sum()

        # calculate sum with no split
        sum = ((data["SalePrice"] - node.const)**2).sum()

        # update feature importance
        featImportance[node.j] += sumL + sumR - sum

        # call recursively on left and right defendants
        self.getImportance(pl, node.leftDescendant, featImportance)
        self.getImportance(pr, node.rightDescendant, featImportance)



class RegressionTreeEnsemble(object):
    def __init__(self, trees=list(), weights=list(), M=0, c=None):
        self.trees = trees
        self.weights = weights
        self.M = M
        self.c = c
        self.trees = trees

    def AddTree(self, tree, weight):
        self.trees.append(tree)
        self.weights.append(weight)

    def SetInitialConstant(self, c):
        self.c = c

    def Evaluate(self, x, m=np.inf):
        res = self.c
        for i in range(min(m, self.M, len(self.trees))):
            res -= self.trees[i].Evaluate(x) * self.weights[i]
        return res
