import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def single_loglikelihood(model, c, t):
    if c is not -1:
        log_likelihood = np.log(sigmoid(model.u[t].T.dot(model.v[c])))
    else:
        loglikelihood = 0
    Nk = model.sample_K_words()
    for nk in Nk:
        log_likelihood += np.log(1 - sigmoid(model.u[t].T.dot(model.v[nk])))
    return log_likelihood


def loglikelihood(data, model):
    # Calculate log likelihood
    log_likelihood = 0
    word_count = 0

    # Get negative samples in a batch to optimize runtime
    batch_Nk = model.sample_batch_words(len(data) * model.largest_sentence * model.hyperParams.C * 2)
    batch_index = 0

    for sentence in data:
        for index, _ in enumerate(sentence):
            t, Wc = model.get_vocabularyIndexAndContext(sentence, index)
            for c in Wc:
                log_likelihood += np.log(sigmoid(model.u[t].T.dot(model.v[c])))
                Nk = batch_Nk[batch_index]
                batch_index += 1
                for nk in Nk:
                    log_likelihood += np.log(1 - sigmoid(model.u[t].T.dot(model.v[nk])))
                word_count += 1

    return log_likelihood / word_count


def minibatch_loglikelihood(minibatch, model):
    # Calculate log likelihood
    log_likelihood = 0
    word_count = 0
    for sample in minibatch:
        t = sample[0]
        for i, c in enumerate(sample[1]):
            log_likelihood += np.log(sigmoid(model.u[t].T.dot(model.v[c])))

            for nk in sample[2][i]:
                log_likelihood += np.log(1 - sigmoid(model.u[t].T.dot(model.v[nk])))
            word_count += 1

    return log_likelihood / word_count
