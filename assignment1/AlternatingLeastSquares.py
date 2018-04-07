import numpy as np

class ALSParameters(object):
    def __init__(self, convergence_threshold):
        super(ALSParameters, self).__init__()
        self.convergence_threshold = convergence_threshold


def LearnModelFromDataUsingALS(data, mfmodel, parameters):
    e = float("inf")
    last_e = float("-Inf")
    M, N = data.shape
    while abs(last_e - e) > parameters.convergence_threshold:

        # Precompute variables
        YTY = mfmodel.u.T.dot(mfmodel.u)
        lambdaI = np.eye(YTY.shape[0]) * mfmodel.lamb.lambda_v
        r = data - mfmodel.mu

        # Update latent variables
        for n in range(N):
            mfmodel.v[n, :] = np.linalg.solve((YTY + lambdaI), r[:, n].dot(mfmodel.u))

        # Precompute variables
        XTX = mfmodel.v.T.dot(mfmodel.v)
        lambdaI = np.eye(XTX.shape[0]) * mfmodel.lamb.lambda_u

        # Update latent variables
        for m in range(M):
            mfmodel.u[m, :] = np.linalg.solve((XTX + lambdaI), r[m, :].dot(mfmodel.v))

        predicted = mfmodel.calc_matrix()
        last_e = e
        e = mfmodel.mean_squared_error(predicted)
        print("Error: %f" % e)

