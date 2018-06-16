
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class RegressionTreeNode(object):
    def __init__(self, j=None, s=None, leftDescendant=None, rightDescendant=None, const=None):
        self.j = j
        self.s = s
        self.leftDescendant = leftDescendant
        self.rightDescendant = rightDescendant
        self.const = const

    def MakeTerminal(self, c):
        self.const = c

    def Split(self, j, s):
        self.j = j
        self.s = s
        self.leftDescendant = RegressionTreeNode()
        self.rightDescendant = RegressionTreeNode()


    def printSubTree(self, tabs):
        if self.const:
            return "\t" * tabs + "return " + str(self.const) + "\n"

        strRep = "\t" * tabs + "if x['" + self.j + "'] <= " + str(self.s) + " then:\n" \
            + self.leftDescendant.printSubTree(tabs + 1) \
            + "\t" * tabs + "if x['" + self.j + "'] > " + str(self.s) + " then:\n" \
            + self.rightDescendant.printSubTree(tabs + 1)
        return strRep


class RegressionTree(object):
    def __init__(self, root):
        self.root = root

    def GetRoot(self):
        return self.root

    def Evaluate(self, x):
        node = self.GetRoot()
        while True:
            if node.const:
                return node.const

            if x[node.j] <= node.s:
                node = node.leftDescendant
            else:
                node = node.rightDescendant



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

    # TODO needs to be implemented
    def Evaluate(self, x, m):
        evaluation = 0

        return evaluation






