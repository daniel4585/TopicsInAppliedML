from Queue import Queue
from RegressionTrees import RegressionTree, RegressionTreeEnsemble
import numpy as np
from CART import CART
from math import log


def calculateLoss(data, ensemble):
    return data.apply(lambda x: (x["SalePrice"] - ensemble.Evaluate(x)) ** 2, axis=1).sum() / data.shape(0)


def GBRT(data, test, M, J, minNodeSize):
    ensemble = RegressionTreeEnsemble(M=M)
    maxDepth = log(J, 2) + 1

    copiedData = data.copy()

    fm = copiedData["SalePrice"].mean()
    for m in range(M):
        print("Generating tree: " + str(m))
        gim = -1 * (data["SalePrice"] - fm)
        copiedData["SalePrice"] = gim
        regressionTree = CART(copiedData, maxDepth, minNodeSize)

        # Calculate weight
        data["SalePrice"] = copiedData.apply(lambda x: regressionTree.Evaluate(x), axis=1)
        bm = (-1.0 * gim * copiedData["SalePrice"]).sum() / (copiedData["SalePrice"] ** 2).sum()
        ensemble.AddTree(regressionTree, bm)

        # Update fm
        fm = fm - bm * copiedData["SalePrice"]

        trainLoss = calculateLoss(data, ensemble)
        testLoss = calculateLoss(test, ensemble)

        print("Train loss: " + str(trainLoss))
        print("Test loss: " + str(testLoss))

    return ensemble





