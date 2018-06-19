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
        strRep = "\t" * tabs + "C: " + str(self.const) + " if x['" + self.j + "'] <= " + str(self.s) + " then:\n" \
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


    """
    this is the recursive call for feature importance it iterates preorder over the tree and passes the featimporatance
    dictionary calculating the split goodness in each inner node 
    
    :param data: the data - is sliced according to node split at each recursive call
    :param node: the current node
    :param featImportance: feature importance memoization dictionary
    :return: updated the featImportance dictionary for all sub tree.
    """
    def getFeatureImprortance(self, data, node, featImportance):
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
        featImportance[node.j] += sum - sumL + sumR

        # call recursively on left and right defendants
        self.getFeatureImprortance(pl, node.leftDescendant, featImportance)
        self.getFeatureImprortance(pr, node.rightDescendant, featImportance)



class RegressionTreeEnsemble(object):
    def __init__(self, M=0):
        self.trees = []
        self.weights = []
        self.M = M
        self.c = 0

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


    """
    Calculates the feature importance of an ensumble of trees according to a given data set

    :param data: the data set
    :return: Normlized and sorted feature importance lsit
    """
    def getFeatureImprortance(self, data):
        featImportance = defaultdict(int)
        featSortedNormlized = []
        mostImportentValue = np.inf

        for i in range(min(self.M, len(self.trees))):
            self.trees[i].getFeatureImprortance(data=data, node=self.trees[i].GetRoot(),featImportance=featImportance)

        # sort and normalize
        for key, value in sorted(featImportance.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            if mostImportentValue == np.inf:
                mostImportentValue = value
            featSortedNormlized.append((key, value / mostImportentValue))
        return featSortedNormlized
