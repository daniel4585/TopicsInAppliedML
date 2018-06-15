from RegressionTrees import RegressionTree


def GetOptimalPartition(x, y):
    return 0,0


def CART(data, maxDepth, minNodeSize):
    X = data.df[:, data.df.columns != 'SalePrice']
    Y = data.df[:, data.df.columns == 'SalePrice']

    d = set()
    for (x, y) in zip(X, Y):
        d.add((x, y))

    d_leaves = set()

    for k in maxDepth:
        d_next = set()
        for (x, y) in d:
            j, s = GetOptimalPartition(x, y)






        d = d_next



    regressionTree = RegressionTree(None)


    return regressionTree

