
import numpy as np
import pandas as pd
from TrainData import TrainData
from ValidationData import ValidationData
from CART import CART, calculateLoss
from GBRT import GBRT


def main():
    df = pd.read_csv("data/train.csv")

    df = df.drop("Id", axis=1)
    df = df[np.isfinite(df["SalePrice"])]

    train, validation = np.split(df.sample(frac=1), [int(.8*len(df))])

    td = TrainData(train)
    vd = ValidationData(validation, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

    #print(pd.DataFrame.info(td.df))

    #regressionTree = CART(td.df, maxDepth=3, minNodeSize=4, numThresholds=10)

    ensemble = GBRT(td.df, vd.df, M=6, J=4, minNodeSize=20, Nu=1.0, Eta=1.0, numThresholds=10)
    for tree in ensemble.trees:
        tree.GetRoot().printSubTree()





    prev = 0
    current = 0
    regressionTree = CART(td.df, maxDepth=3, minNodeSize=4, numThresholds=10)
    current = calculateLoss(td.df, regressionTree)
    print "train loss 3 " + str(current) + " diff = " + str(current - prev)
    print regressionTree
    regressionTree.getFeatureImprortance(td.df)
    prev = current

    regressionTree = CART(td.df, maxDepth=4, minNodeSize=4, numThresholds=10)
    current = calculateLoss(td.df, regressionTree)
    print "train loss 5 " + str(current) + " diff = " + str(current - prev)
    print regressionTree
    regressionTree.getFeatureImprortance(td.df)
    prev = current

    regressionTree = CART(td.df, maxDepth=5, minNodeSize=4, numThresholds=10)
    current = calculateLoss(td.df, regressionTree)
    print "train loss 7 " + str(current) + " diff = " + str(current - prev)
    print regressionTree
    regressionTree.getFeatureImprortance(td.df)
    prev = current




if __name__ == '__main__':
    main()
