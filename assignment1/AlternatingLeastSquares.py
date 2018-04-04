import numpy as np


class Parameters(object, ):
    def __init__(self, convergence_threshold):
        super(Parameters, self).__init__()
        self.convergence_threshold = convergence_threshold



def LearnModelFromDataUsingALS(data, mfmodel, parameters):

    e = float("inf")
    while e > parameters.convergence_threshold:
        xs, ys = data.nonzero()
        for x, y in zip(xs, ys):
            sample = (x, y, data[x, y])
            alternating_least_squares_update(sample, mfmodel, parameters)

        predicted = mfmodel.calc_matrix()
        e = mfmodel.mean_squared_error(predicted)
        print("Error: %f" % e)



def alternating_least_squares_update(sample, mfmodel, parameters):



def calc_matrix(mfmodel):
    return (mfmodel.mu + mfmodel.b_n[:, np.newaxis] + mfmodel.b_m[np.newaxis, :]).T + mfmodel.u.dot(mfmodel.v.T)


def mean_squared_error(data, predicted):
    xs, ys = data.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        error += pow(data[x, y] - predicted[x, y], 2)
    return np.sqrt(error)
