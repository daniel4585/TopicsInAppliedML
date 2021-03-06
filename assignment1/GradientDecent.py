import numpy as np
import os, errno

from MFModel import MFModel
from utils import mean_squared_error
from utils import write_error_to_file


class SGDParameters(object):
    def __init__(self, steps, alpha):
        super(SGDParameters, self).__init__()
        self.steps = steps
        self.alpha = alpha


def LearnModelFromDataUsingSGD(data, mfmodel, parameters, extra_data_set=None):
    print("Training model using SGD")

    try:
        os.remove("output/SGD_train_error.txt")
        os.remove("output/SGD_test_error.txt")
    except OSError:
        pass

    for step in range(parameters.steps):

        predicted = mfmodel.calc_matrix()
        print("Step: %s, error: %f" % (step, mean_squared_error(mfmodel, predicted, data)))

        write_error_to_file(mfmodel, predicted, data, "SGD_train_error.txt")
        if extra_data_set is not None:
            write_error_to_file(mfmodel, predicted, extra_data_set, "SGD_test_error.txt")

        xs, ys = data.nonzero()
        for x, y in zip(xs, ys):
            sample = (x, y, data[x, y])
            gradient_decent_update(sample, mfmodel, parameters)


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

