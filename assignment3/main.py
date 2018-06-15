import pandas as pd
import numpy as np

def main():
    train = pd.read_csv("data/train.csv")
    train = train.drop("Id", axis=1)
    #print(pd.DataFrame.describe(train))
    print(pd.DataFrame.info(train))
    train = train[np.isfinite(train["SalePrice"])]
    print(pd.DataFrame.info(train))






if __name__ == '__main__':
    main()
