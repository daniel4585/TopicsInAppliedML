import pickle
import numpy as np
import pandas as pd
from TrainData import TrainData
from ValidationData import ValidationData
from CART import CART, calculateLoss
from GBRT import GBRT

def pickle_load(path):
    with open(path, 'rb') as input:
        return pickle.load(input)


def pickle_save(path, value):
    with open(path, 'wb') as output:
        pickle.dump(value, output, pickle.HIGHEST_PROTOCOL)


def main():
    df = pd.read_csv("data/train.csv")

    df = df.drop("Id", axis=1)
    df = df[np.isfinite(df["SalePrice"])]

    train, validation = np.split(df.sample(frac=1), [int(.8*len(df))])

    td = TrainData(train)
    vd = ValidationData(validation, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

    #print(pd.DataFrame.info(td.df))
    #regressionTree = CART(td.df, maxDepth=3, minNodeSize=4, numThresholds=10)

    ensemble = GBRT(td.df, vd.df, M=100, J=16, minNodeSize=5, Nu=1.0, Eta=0.5)
    for tree in ensemble.trees:
        tree.GetRoot().printSubTree()

    pickle_save('ensemble.pkl', ensemble)



if __name__ == '__main__':
    main()
