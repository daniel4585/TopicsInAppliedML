from SGD import sigmoid


def loglikelihood(data, model):
    # Calculate log likelihood
    log_likelihood = 0
    for sentence in data:
        for word in sentence:
            t, Wc = model.get_vocabularyAndContext(sentence, word)
            for i, c in Wc:
                log_likelihood += np.log(sigmoid(model.u[t].T.dot(model.v[c])))

                Nk = model.sample_K_words()
                for nk in Nk:
                    log_likelihood += np.log(1 - sigmoid(model.u[t].T.dot(model.v[nk])))

    return log_likelihood