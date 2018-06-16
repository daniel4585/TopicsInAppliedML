from Queue import Queue
from RegressionTrees import RegressionTree
import numpy as np

from RegressionTrees import RegressionTreeNode


def GetOptimalPartition(data, p):
    for j in data.df:
        s_j = data.df[j].unique()
        min_l = np.inf
        min_s = 0
        min_j = 0
        min_cl = 0
        min_cr = 0
        for s in s_j:
            R_lt = data.df.loc[data.df[j] <= s]
            R_gt = data.df.loc[data.df[j] > s]
            c_lt = (1/data.df[R_lt].shape(0)) * data.df[R_lt, "SalePrice"].sum()
            c_gt = (1/data.df[R_gt].shape(0)) * data.df[R_gt, "SalePrice"].sum()
            l = 0
            for v in data.df[R_lt, "SalePrice"]:
                l += (v - c_lt)**2
            for v in data.df[R_gt, "SalePrice"]:
                l += (v - c_gt) ** 2
            if l < min_l:
                min_l = l
                min_s = s
                min_j = j
                min_cl = c_lt
                min_cr = c_gt
    return min_s, min_j, min_cl, min_cr


def CART(data, maxDepth, minNodeSize):
    q = Queue()
    root = RegressionTreeNode()
    q.put((root, data.df.loc[:, :]))
    for k in maxDepth:
        while q.qsize() is not 0:
            node, p ,c = q.get()
            j, s, cl, cr = GetOptimalPartition(data, p)
            pl = data.df.loc[data.df[j] <= s]
            pr = data.df.loc[data.df[j] > s]
            if len(pl) >= minNodeSize and len(pr) >= minNodeSize:
                node.Split(s, j)
                q.put((node.leftDescendant, pl, cl))
                q.put((node.rightDescendant, pr, cr))
            else:
                node.MakeTerminal(c)

    return RegressionTree(root)

