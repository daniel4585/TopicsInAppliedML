import numpy as np

class SGD(object):

    def fit(self, modelParameters):

        toAnneal = modelParameters.hyperParams.annealingRate
        eta = modelParameters.hyperParams.eta
        for _ in range(modelParameters.hyperParams.iterations):

            # Initialize mini-batch gradients
            gT = np.zeros((len(modelParameters.vocabulary), modelParameters.hyperParamsD))
            gC = np.zeros((len(modelParameters.vocabulary), modelParameters.hyperParamsD))

            for _ in range(modelParameters.hyperParams.minibatchsize):
                # Sample a word from the data set
                t = 5
                Wc = [1, 10, 50, 100, 200]

                for c in Wc:
                    # Positive sample
                    sig = sigmoid(v[c] * u[t])
                    gT[t] = gT[t] + (1 - sig*v[nk])*v[c]
                    gC[nk] = gC[nk] + (1 - sig*v[nk])*u[t]

                    # Negative samples
                    Nk = [2, 1203, 32, 24]
                    for nk in Nk:
                        sig = sigmoid(v[nk]*u[t]
                        gT[t] = gT[t] - sig*v[nk]
                        gC[nk] = gC[nk] - sig*u[t]

            u = u + eta


            toAnneal -= 1
            if toAnneal == 0:
                toAnneal = modelParameters.hyperParams.annealingRate
                eta = eta / 2.0




