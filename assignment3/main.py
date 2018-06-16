
import numpy as np
import pandas as pd
from TrainData import TrainData
from ValidationData import ValidationData
from CART import CART
from GBRT import GBRT


def main():
    df = pd.read_csv("data/train.csv")

    df = df.drop("Id", axis=1)
    df = df[np.isfinite(df["SalePrice"])]

    train, validation = np.split(df.sample(frac=1), [int(.8*len(df))])

    td = TrainData(train)
    vd = ValidationData(validation, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

    #print(pd.DataFrame.info(td.df))
    #regressionTree = CART(td.df, maxDepth=10, minNodeSize=4)
    #regressionTree.GetRoot().printSubTree()

    ensemble = GBRT(td.df, vd.df, M=6, J=4, minNodeSize=20, Nu=1.0, Eta=1.0)
    for tree in ensemble.trees:
        tree.GetRoot().printSubTree()



if __name__ == '__main__':
    main()
