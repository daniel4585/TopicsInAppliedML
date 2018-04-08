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
        u_with_bias = np.concatenate((mfmodel.u, np.ones((M, 1))), axis=1)
        v_with_bias = np.concatenate((mfmodel.v, np.expand_dims(mfmodel.b_n, axis=1)), axis=1)

        YTY = u_with_bias.T.dot(u_with_bias)
        lambdaI = np.eye(YTY.shape[0]) * mfmodel.lamb.lambda_v
        r = (data.T - mfmodel.b_m).T - mfmodel.mu

        # Update latent variables
        for n in range(N):
            v_with_bias[n, :] = np.linalg.solve((YTY + lambdaI), r[:, n].dot(u_with_bias))

        # Extract variables and bias
        mfmodel.v = v_with_bias[:, :-1]
        mfmodel.b_n = v_with_bias[:, -1]


        # Precompute variables
        u_with_bias = np.concatenate((mfmodel.u, np.expand_dims(mfmodel.b_m, axis=1)), axis=1)
        v_with_bias = np.concatenate((mfmodel.v, np.ones((N, 1))), axis=1)

        XTX = v_with_bias.T.dot(v_with_bias)
        lambdaI = np.eye(XTX.shape[0]) * mfmodel.lamb.lambda_u
        r = data - mfmodel.mu - mfmodel.b_n

        # Update latent variables
        for m in range(M):
            u_with_bias[m, :] = np.linalg.solve((XTX + lambdaI), r[m, :].dot(v_with_bias))

        # Extract variables and bias
        mfmodel.u = u_with_bias[:, :-1]
        mfmodel.b_m = u_with_bias[:, -1]

        # Summarize step
        predicted = mfmodel.calc_matrix()
        last_e = e
        e = mfmodel.mean_squared_error(predicted)
        print("Error: %f" % e)

