import numpy as np
import time

from Diagnostics import minibatch_loglikelihood, loglikelihood, sigmoid
import os


class SGD(object):

    def LearnParamsUsingSGD(self, trainTest, model):
        train = trainTest.train
        test = trainTest.test

        toAnneal = model.hyperParams.annealingRate
        eta = model.hyperParams.eta

        output_file = "output/loglikelihood.txt"
        try:
            os.remove(output_file)
        except OSError:
            pass

        # Get negative samples in a batch to optimize runtime
        batch_Nk = model.sample_batch_words(model.hyperParams.iterations * model.hyperParams.minibatchsize * model.hyperParams.C * 2)
        batch_index = 0

        for iter in range(model.hyperParams.iterations):

            # Initialize mini-batch gradients
            #gT = np.zeros((len(model.vocabulary), model.hyperParams.D))
            #gC = np.zeros((len(model.vocabulary), model.hyperParams.D))

            gT = {}
            gC = {}

            # Save mini-batch for calculating log-likelihood
            minibatch = []

            #start = time.time()
            for _ in range(model.hyperParams.minibatchsize):
                # Sample a word from the data set
                t, Wc = model.sample_target_context(train)

                minibatch.append((t, Wc, []))

                for c in Wc:
                    # Positive sample
                    sig = sigmoid(model.v[c].T.dot(model.u[t]))
                    if t not in gT:
                        gT[t] = (1 - sig) * model.v[c]
                    else:
                        gT[t] = gT[t] + (1 - sig) * model.v[c]

                    if c not in gC:
                        gC[c] = (1 - sig) * model.u[t]
                    else:
                        gC[c] = gC[c] + (1 - sig) * model.u[t]

                    # Negative samples
                    Nk = batch_Nk[batch_index]
                    batch_index += 1
                    for nk in Nk:
                        sig = sigmoid(model.v[nk].T.dot(model.u[t]))
                        if t not in gT:
                            gT[t] = len(Wc) * sig * model.v[nk]
                        else:
                            gT[t] = gT[t] - len(Wc) * sig * model.v[nk]

                        if nk not in gC:
                            gC[nk] = len(Wc) * sig * model.u[t]
                        else:
                            gC[nk] = gC[nk] - len(Wc) * sig * model.u[t]

                    minibatch[-1][2].append(Nk)

            #print(time.time() - start)

            # Update model matrices
            #model.u = model.u + eta * gT / model.hyperParams.minibatchsize
            #model.v = model.v + eta * gC / model.hyperParams.minibatchsize

            for key, val in gT.items():
                model.u[key] = model.u[key] + eta * val / model.hyperParams.minibatchsize

            for key, val in gC.items():
                model.v[key] = model.v[key] + eta * val / model.hyperParams.minibatchsize

            model.normalize()

            # Update learning rate
            toAnneal -= 1
            if toAnneal == 0:
                toAnneal = model.hyperParams.annealingRate
                eta = eta / 2.0

            # Calculate log likelihood
            minibatch_log_likelihood = minibatch_loglikelihood(minibatch, model)
            #print("Iteration: " + str(iter) + " Log likelihood: " + str(minibatch_log_likelihood))

            # Print loglikelihood to file
            if iter % model.hyperParams.X == 0:
                with open(output_file, "a") as output:
                    output.write("Iteration: " + str(iter) + "\n")
                    output.write("Minibatch loglikelihood: " + str(minibatch_log_likelihood) + "\n")
                    output.write("Loglikelihood: " + str(loglikelihood(test, model)) + "\n")










