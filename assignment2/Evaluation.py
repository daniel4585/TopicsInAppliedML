import heapq
import matplotlib.pyplot as plt

from Diagnostics import *


def PredictContext(model, wt):
    min_heap = []
    t = model.vocabulary[wt][1]
    for key, value in model.vocabulary.items():
        c = value[1]
        contextLikelihood = single_loglikelihood(model, c, t)
        if len(min_heap) < 10:
            heapq.heappush(min_heap, (contextLikelihood, key))
        else:
            heapq.heappushpop(min_heap, (contextLikelihood, key))

    return sorted(min_heap, reverse=True)


def PredictInput(model, wcList):
    min_heap = []
    for key, value in model.vocabulary.items():
        t = value[1]
        contextLikelihood = 0
        for wc in wcList:
            vocabVal = model.vocabulary[wc]
            if vocabVal is 0:
               c = 0
            else:
                c = vocabVal[1]
            contextLikelihood += single_loglikelihood(model, c, t)
        if len(min_heap) < 10:
            heapq.heappush(min_heap, (contextLikelihood, key))
        else:
            heapq.heappushpop(min_heap, (contextLikelihood, key))

    return sorted(min_heap, reverse=True)


def ScatterMatrix(model, words):
    indices = [model.vocabulary[word][1] for word in words]
    vecs_u = model.u[indices]
    x_u = vecs_u[:, 0]
    y_u = vecs_u[:, 1]
    vecs_v = model.v[indices]
    x_v = vecs_v[:, 0]
    y_v = vecs_v[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x_u, y_u, color='r', label="U")
    ax.scatter(x_v, y_v, color='b', label="V")
    ax.legend()
    for i, txt in enumerate(words):
        ax.annotate(txt, (x_u[i], y_u[i]))
        ax.annotate(txt, (x_v[i], y_v[i]))
    plt.suptitle("Model HyperParams:" + str(model.hyperParams))
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
    return sorted(min_heap, reverse=True)


