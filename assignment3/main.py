
import numpy as np
import pandas as pd
from TrainData import TrainData
from ValidationData import ValidationData
from CART import CART


def main():
    df = pd.read_csv("data/train.csv")
    print(pd.DataFrame.info(df))

    df = df.drop("Id", axis=1)
    df = df[np.isfinite(df["SalePrice"])]

    train, validation = np.split(df.sample(frac=1), [int(.8*len(df))])

    td = TrainData(train)
    vd = ValidationData(validation, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

    regressionTree = CART(td, maxDepth=2, minNodeSize=20)
    print(regressionTree.GetRoot().printSubTree(0))




if __name__ == '__main__':
    main()
