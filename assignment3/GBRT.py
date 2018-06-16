from Queue import Queue
from RegressionTrees import RegressionTree, RegressionTreeEnsemble
import numpy as np
from CART import CART
from math import log


def calculateLoss(data, ensemble):
    return data.apply(lambda x: (x["SalePrice"] - ensemble.Evaluate(x)) ** 2, axis=1).sum() / data.shape[0]


def GBRT(data, test, M, J, minNodeSize, Nu=1.0, Eta=1.0, numThresholds=np.inf):
    ensemble = RegressionTreeEnsemble(M=M)
    maxDepth = int(log(J, 2))

    copiedData = data.copy()
    y = data["SalePrice"]

    fm = copiedData["SalePrice"].mean()
    for m in range(M):
        print("Generating tree: " + str(m))
        gim = -1 * (y - fm)
        copiedData["SalePrice"] = gim

        # Subsampling
        subsampled = copiedData.sample(frac=Eta)
        regressionTree = CART(subsampled, maxDepth, minNodeSize, numThresholds)

        # Calculate weight
        phi = copiedData.apply(lambda x: regressionTree.Evaluate(x), axis=1)
        bm = (gim * phi).sum() / (phi ** 2).sum()
        print(bm)
        ensemble.AddTree(regressionTree, bm)

        # Update fm
        fm = fm - Nu * bm * phi

        trainLoss = calculateLoss(data, ensemble)
        testLoss = calculateLoss(test, ensemble)

        print("Train loss: " + str(trainLoss))
        print("Test loss: " + str(testLoss))

    return ensemble





