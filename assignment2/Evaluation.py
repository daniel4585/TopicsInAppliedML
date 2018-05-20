import heapq
import matplotlib.pyplot as plt

from Diagnostics import *
def PredictContext(model,wt):
    min_heap = []
    t = model.vocabulary[wt][1]
    for key, value in model.vocabulary.items():
        c = value[1]
        contextLikelihood = single_loglikelihood(model, c, t)
        if len(min_heap) < 10:
            heapq.heappush(min_heap, (contextLikelihood, key))
        else:
            heapq.heappushpop(min_heap, (contextLikelihood, key))

    return min_heap

def PredictInput(model, wcList):
    min_heap = []
    for key, value in model.vocabulary.items():
        t = value[1]
        contextLikelihood = 0
        for wc in wcList:
            c = model.vocabulary[wc][1]
            contextLikelihood += single_loglikelihood(model, c, t)
        if len(min_heap) < 10:
            heapq.heappush(min_heap, (contextLikelihood, key))
        else:
            heapq.heappushpop(min_heap, (contextLikelihood, key))

    return min_heap

def ScatterMatrix(model, words):
    indices = model.vocabulary[words][1]
    x, y = model.u[indices][0:2]
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(words):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def AnalogySolver(model, w1, w2, w3):
    min_heap = []
    u1 = model.u[model.vocabulary[w1][1]]
    u2 = model.u[model.vocabulary[w2][1]]
    u3 = model.u[model.vocabulary[w3][1]]

    for key, value in model.vocabulary.items():
        t = value[1]
        ut = model.u[t]
        val = ut.T.dot(u1-u2+u3)
        if len(min_heap) < 10:
            heapq.heappush(min_heap, (val, key))
        else:
            heapq.heappushpop(min_heap, (val, key))



