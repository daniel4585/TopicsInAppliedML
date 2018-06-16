from Queue import Queue
from RegressionTrees import RegressionTree
import numpy as np

from RegressionTrees import RegressionTreeNode


def GetOptimalPartition(p):
    min_l = np.inf
    min_s = 0
    min_j = 0
    min_cl = 0
    min_cr = 0
    for j in p.drop("SalePrice", axis=1):
        s_j = p[j].unique()
        for s in s_j:
            R_lt = p.loc[p[j] <= s]
            R_gt = p.loc[p[j] > s]
            c_lt = 0 if R_lt.shape[0] == 0 else (1.0/R_lt.shape[0]) * R_lt["SalePrice"].sum()
            c_gt = 0 if R_gt.shape[0] == 0 else (1.0/R_gt.shape[0]) * R_gt["SalePrice"].sum()
            l = 0
            l += ((R_lt["SalePrice"] - c_lt) ** 2).sum()
            l += ((R_gt["SalePrice"] - c_gt) ** 2).sum()
            if l < min_l:
                min_l = l
                min_s = s
                min_j = j
                min_cl = c_lt
                min_cr = c_gt
    return min_j, min_s, min_cl, min_cr


def CART(data, maxDepth, minNodeSize):
    if maxDepth == 0:
        print "Max depth must be greater than 1"
        return None

    q = Queue()
    root = RegressionTreeNode()
    q.put((root, data, np.inf, 0))
    while q.qsize() != 0:
        node, p, c, level = q.get()
        if level == maxDepth:
            node.MakeTerminal(c)
            continue

        j, s, cl, cr = GetOptimalPartition(p)
        pl = p.loc[p[j] <= s]
        pr = p.loc[p[j] > s]
        if len(pl) >= minNodeSize and len(pr) >= minNodeSize:
            node.Split(j, s)
            q.put((node.leftDescendant, pl, cl, level + 1))
            q.put((node.rightDescendant, pr, cr, level + 1))
        else:
            node.MakeTerminal(c)

    while q.qsize() != 0:
        node, _, c = q.get()
        node.MakeTerminal(c)

    return RegressionTree(root)

