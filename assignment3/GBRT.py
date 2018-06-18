from Queue import Queue
from RegressionTrees import RegressionTree, RegressionTreeEnsemble, RegressionTreeNode
import numpy as np
from CART import CART
from math import log
import os

def calculateLoss(data, ensemble):
    return data.apply(lambda x: (x["SalePrice"] - ensemble.Evaluate(x)) ** 2, axis=1).mean()

def GBRT(data, test, hyperparams, outputFile="results.txt"):
    ensamble, _ , _ = GBRT_WithLoss(data, test, hyperparams, outputFile)
def GBRT_WithLoss(data, test, hyperparams, outputFile="results.txt"):
    ensemble = RegressionTreeEnsemble(M=hyperparams.numOfTrees)
    maxDepth = int(log(hyperparams.maxDepth, 2))

    try:
        os.remove("output/" + outputFile)
    except OSError, IOError:
        pass

    copiedData = data.copy(deep="all")
    y = data["SalePrice"]
    ensemble.SetInitialConstant(y.mean())

    fm = copiedData["SalePrice"].mean()
    for m in range(hyperparams.numOfTrees):
        print("Generating tree: " + str(m))
        gim = -1 * (y - fm)
        copiedData["SalePrice"] = gim

        # Subsampling and fit CART
        subsampled = copiedData.sample(frac=hyperparams.eta)
        regressionTree = CART(subsampled, maxDepth, hyperparams.minNodeSize, hyperparams.numThresholds)

        # Calculate weight
        phi = copiedData.apply(regressionTree.Evaluate, axis=1)
        bm = np.dot(gim, phi) / np.sum(np.power(phi, 2))
        ensemble.AddTree(regressionTree, hyperparams.nu * bm)
        print("Bm " + str(bm))

        # Update fm
        fm = copiedData.apply(ensemble.Evaluate, axis=1)

        trainLoss = calculateLoss(data, ensemble)
        testLoss = calculateLoss(test, ensemble)

        print("Train loss: " + str(trainLoss))
        print("Test loss: " + str(testLoss))

        with open("output/" + outputFile, 'a') as output:
            output.write("Train Loss: " + str(trainLoss) + "\n")
            output.write("Test Loss: " + str(testLoss) + "\n")


    return ensemble, trainLoss, testLoss





