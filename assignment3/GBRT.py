from Queue import Queue
from RegressionTrees import RegressionTree, RegressionTreeEnsemble, RegressionTreeNode
import numpy as np
from CART import CART
from math import log


def calculateLoss(data, ensemble):
    return data.apply(lambda x: (x["SalePrice"] - ensemble.Evaluate(x)) ** 2, axis=1).mean()


def GBRT(data, test, M, J, minNodeSize, Nu=1.0, Eta=1.0, numThresholds=np.inf):
    ensemble = RegressionTreeEnsemble(M=M)
    maxDepth = int(log(J, 2))

    try:
        os.remove("output/results.txt")
    except OSError, IOError:
        pass

    copiedData = data.copy()
    y = data["SalePrice"]
    ensemble.SetInitialConstant(y.mean())

    fm = copiedData["SalePrice"].mean()
    for m in range(M):
        print("Generating tree: " + str(m))
        gim = -1 * (y - fm)
        copiedData["SalePrice"] = gim

        # Subsampling and fit CART
        subsampled = copiedData.sample(frac=Eta)
        regressionTree = CART(subsampled, maxDepth, minNodeSize, numThresholds)

        # Calculate weight
        phi = copiedData.apply(regressionTree.Evaluate, axis=1)
        bm = np.dot(gim, phi) / np.sum(np.power(phi, 2))
        ensemble.AddTree(regressionTree, Nu * bm)
        print("Bm " + str(bm))

        # Update fm
        fm = copiedData.apply(ensemble.Evaluate, axis=1)

        trainLoss = calculateLoss(data, ensemble)
        testLoss = calculateLoss(test, ensemble)

        print("Train loss: " + str(trainLoss))
        print("Test loss: " + str(testLoss))

        with open("output/results.txt", 'a') as output:
            output.write("Train Loss: " + str(trainLoss) + "\n")
            output.write("Test Loss: " + str(testLoss) + "\n")


    return ensemble





