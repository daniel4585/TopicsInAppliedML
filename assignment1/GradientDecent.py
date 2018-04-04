import numpy as np

from MFModel import MFModel


class Parameters(object):
    def __init__(self, steps, alpha):
        super(Parameters, self).__init__()
        self.steps = steps
        self.alpha = alpha


def LearnModelFromDataUsingSGD(data, mfmodel, parameters):

    for step in range(parameters.steps):
        xs, ys = data.nonzero()
        for x, y in zip(xs, ys):
            sample = (x, y, data[x, y])
            gradient_decent_update(sample, mfmodel, parameters)

        predicted = mfmodel.calc_matrix()
        print("Step: %s, error: %f" % (step, mfmodel.mean_squared_error(predicted)))


def gradient_decent_update(sample, mfmodel, parameters):
    i = sample[0]
    j = sample[1]

    # Calculate error
    prediction = mfmodel.mu + mfmodel.b_m[i] + mfmodel.b_n[j] + np.dot(mfmodel.u[i, :], mfmodel.v[j, :])
    e = sample[2] - prediction

    # Update user and movie latent feature matrices
    mfmodel.u[i, :] -= parameters.alpha * (-1 * e * mfmodel.v[j, :] + mfmodel.lamb.lambda_u * mfmodel.u[i, :])
    mfmodel.v[j, :] -= parameters.alpha * (-1 * e * mfmodel.u[i, :] + mfmodel.lamb.lambda_v * mfmodel.v[j, :])

    # Update biases
    mfmodel.b_m[i] -= parameters.alpha * (-1 * e + mfmodel.lamb.lambda_b_u * mfmodel.b_m[i])
    mfmodel.b_n[j] -= parameters.alpha * (-1 * e + mfmodel.lamb.lambda_b_v * mfmodel.b_n[j])

    if mfmodel.b_m[i] > 200 or mfmodel.b_m[i] < -200:
        print(mfmodel.b_m[i])

    if e > 200 or e < -200:
        print(sample)
        print(e)



