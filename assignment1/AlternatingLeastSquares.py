import numpy as np


class ALSParameters(object):
    def __init__(self, convergence_threshold, lamb):
        super(ALSParameters, self).__init__()
        self.convergence_threshold = convergence_threshold
        self.lamb = lamb



def  LearnModelFromDataUsingALS(data, mfmodel, parameters):

    e = float("inf")
    M , N  = data.shape
    while e > parameters.convergence_threshold:
        for n in range(N-1):
            alternating_least_squares_update_vn(n, data, mfmodel, parameters)
        for m in range(M-1):
            alternating_least_squares_update_um(m, data, mfmodel, parameters)
        predicted = mfmodel.calc_matrix()
        e = mfmodel.mean_squared_error(predicted)
        print("Error: %f" % e)



def  alternating_least_squares_update_vn(n,data, mfmodel, parameters):

    y = data[:, n]
    X = mfmodel.u.T

    # print 'y: {0} X: {1}'.format(y.shape, X.shape)
    # print 'inv: {0}'.format(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).shape)
    # print "X'y: {0}".format(X.dot(y).shape)
    # print 'full: {0}'.format(np.squeeze(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).dot(X.dot(y))).shape)

    # Closed-form solution for the j'th ridge regression
    mfmodel.v[n, :] = np.squeeze(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).dot(X.dot(y)))


    # r_nb = data[:,n] - mfmodel.b_m - mfmodel.mu - mfmodel.b_n[n]
    # # nonzeros = data[:, n].nonzero()[0]
    # X = mfmodel.u[:,n].T
    # y_n = mfmodel.v[:,n]
    # tmp = X.T.dot(X) + parameters.lamb * np.eye(X.shape[0])
    # tmp = np.linalg.inv(tmp)
    # tmp2 = X.T.dot(r_nb)
    # y_n = tmp.dot(tmp2)
    # mfmodel.v[:, n] = y_n

def  alternating_least_squares_update_um(m, data, mfmodel, parameters):
    y = data[m, :]
    X = mfmodel.v.T

    # print 'y: {0} X: {1}'.format(y.shape, X.shape)
    # print 'inv: {0}'.format(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).shape)
    # print "X'y: {0}".format(X.dot(y).shape)
    # print 'full: {0}'.format(
    # np.squeeze(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).dot(X.dot(y))).shape)

    # Closed-form solution for the j'th ridge regression
    mfmodel.u[m, :] = np.squeeze(np.linalg.inv(X.dot(X.T) + parameters.lamb * np.eye(X.shape[0])).dot(X.dot(y)))

    # r_mb = data[:,m] - mfmodel.b_n - mfmodel.mu - mfmodel.b_m[m]
    # # nonzeros = data[:, n].nonzero()[0]
    # Y =  mfmodel.v[:, m].T
    # X_m = mfmodel.u[:, m]
    # X_m = np.linalg.inv(Y.T.dot(Y) + parameters.lamb * np.eye(Y.shape[1])).dot(Y.T.dot(r_mb))
    # mfmodel.u[:, m] = X_m

def  calc_matrix(mfmodel):
    return (mfmodel.mu + mfmodel.b_n[:, np.newaxis] + mfmodel.b_m[np.newaxis, :]).T + mfmodel.u.dot(mfmodel.v.T)


def mean_squared_error(data, predicted):
    xs, ys = data.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        error += pow(data[x, y] - predicted[x, y], 2)
    return np.sqrt(error)
