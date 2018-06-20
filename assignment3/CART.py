from Queue import Queue
from RegressionTrees import RegressionTree
import numpy as np

from RegressionTrees import RegressionTreeNode
import target_global


def GetOptimalPartition(p, numThresholds=np.inf):
    min_l = np.inf
    min_s = 0
    min_j = 0
    min_cl = 0
    min_cr = 0
    for j in p.drop(target_global.target_name, axis=1):
        precentiles = np.linspace(0.0, 1.0, min(numThresholds, p[j].nunique()), False)
        s_j = [p[j].quantile(x) for x in precentiles[1:]]
        for s in s_j:
            R_lt = p.loc[p[j] <= s]
            R_gt = p.loc[p[j] > s]
            c_lt = 0 if R_lt.shape[0] == 0 else (1.0/R_lt.shape[0]) * R_lt[target_global.target_name].sum()
            c_gt = 0 if R_gt.shape[0] == 0 else (1.0/R_gt.shape[0]) * R_gt[target_global.target_name].sum()
            l = 0
            l += ((R_lt[target_global.target_name] - c_lt) ** 2).sum()
            l += ((R_gt[target_global.target_name] - c_gt) ** 2).sum()
            if l < min_l:
                min_l = l
                min_s = s
                min_j = j
                min_cl = c_lt
                min_cr = c_gt
    return min_j, min_s, min_cl, min_cr


def CART(data, maxDepth, minNodeSize, numThresholds):
    q = Queue()

    c_root = data[target_global.target_name].mean()
    root = RegressionTreeNode(const=c_root)
    q.put((root, data, 0))
    while q.qsize() != 0:
        node, p, level = q.get()
        if level == maxDepth:
            continue

        j, s, cl, cr = GetOptimalPartition(p, numThresholds)
        pl = p.loc[p[j] <= s]
        pr = p.loc[p[j] > s]
        if len(pl) >= minNodeSize and len(pr) >= minNodeSize:
            node.Split(j, s, cl, cr)
            q.put((node.leftDescendant, pl, level + 1))
            q.put((node.rightDescendant, pr, level + 1))

    while q.qsize() != 0:
        node, _, c = q.get()
        node.MakeTerminal(c)

    return RegressionTree(root)

def calculateLoss(data, tree):
    return data.apply(lambda x: (x[target_global.target_name] - tree.Evaluate(x)) ** 2, axis=1).sum() / data.shape[0]

