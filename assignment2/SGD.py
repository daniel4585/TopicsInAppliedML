import numpy as np
from Diagnostics import minibatch_loglikelihood, loglikelihood, sigmoid
import os


class SGD(object):

    def LearnParamsUsingSGD(self, train, test, model):

        toAnneal = model.hyperParams.annealingRate
        eta = model.hyperParams.eta

        output_file = "output/loglikelihood.txt"
        try:
            os.remove(output_file)
        except OSError:
            pass

        for iter in range(model.hyperParams.iterations):
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

            # Update model matrices
            model.u = u + eta * gT
            model.v = v + eta * gC
            model.normalize()

            # Update learning rate
            toAnneal -= 1
            if toAnneal == 0:
                toAnneal = model.hyperParams.annealingRate
                eta = eta / 2.0

            # Calculate log likelihood
            minibatch_log_likelihood = minibatch_loglikelihood(minibatch, model)
            print("Iteration: " + str(iter) + " Log likelihood: " + str(minibatch_log_likelihood))

            # Print loglikelihood to file
            if iter % model.hyperParams.X == 0:
                with open(output_file, "a") as output:
                    output.write("Iteration: " + str(iter) + "\n")
                    output.write("Minibatch loglikelihood: " + str(minibatch_log_likelihood))
                    output.write("Loglikelihood: " + str(loglikelihood(test, model)))










