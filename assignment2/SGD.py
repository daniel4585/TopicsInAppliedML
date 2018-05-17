import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class SGD(object):

    def LearnParamsUsingSGD(self, train, model):

        toAnneal = model.hyperParams.annealingRate
        eta = model.hyperParams.eta
        for _ in range(model.hyperParams.iterations):
            u = model.u
            v = model.v

            # Initialize mini-batch gradients
            gT = np.zeros((len(model.vocabulary), model.hyperParams.D))
            gC = np.zeros((len(model.vocabulary), model.hyperParams.D))

            # Save mini-batch for calculating log-likelihood
            minibatch = []

            for _ in range(model.hyperParams.minibatchsize):
                # Sample a word from the data set
                t, Wc = model.sample_target_context(train)

                minibatch.append((t, Wc, []))

                for c in Wc:
                    # Positive sample
                    sig = sigmoid(v[c].T.dot(u[t]))
                    gT[t] = gT[t] + (1 - sig) * v[c]
                    gC[c] = gC[c] + (1 - sig) * u[t]

                    # Negative samples
                    Nk = model.sample_K_words()
                    for nk in Nk:
                        sig = sigmoid(v[nk].T.dot(u[t]))
                        gT[t] = gT[t] - sig * v[nk]
                        gC[nk] = gC[nk] - sig * u[t]

                    minibatch[-1][2].append(Nk)

            model.u = u + eta * gT
            model.v = v + eta * gC
            model.normalize()

            toAnneal -= 1
            if toAnneal == 0:
                toAnneal = model.hyperParams.annealingRate
                eta = eta / 2.0

            log_likelihood = 0
            for sample in minibatch:
                t = sample[0]
                for i, c in enumerate(sample[1]):
                    log_likelihood += math.log10(sigmoid(model.u[t].T.dot(model.v[c])))

                    for nk in sample[2][i]:
                        log_likelihood -= math.log10(1 - sigmoid(model.u[t].T.dot(model.v[nk])))

            print("Log likelihood: " + str(log_likelihood))







