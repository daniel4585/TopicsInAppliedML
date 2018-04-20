import numpy as np
import os, errno

from utils import write_error_to_file, mean_squared_error


class ALSParameters(object):
    def __init__(self, convergence_threshold):
        super(ALSParameters, self).__init__()
        self.convergence_threshold = convergence_threshold


def LearnModelFromDataUsingALS(data, mfmodel, parameters, extra_data_set=None):
    print("Training model using ALS")

    e = float("inf")
    last_e = float("-Inf")
    M, N = data.shape

    try:
        os.remove("output/ALS_train_error.txt")
        os.remove("output/ALS_test_error.txt")
    except OSError:
        pass

    while abs(last_e - e) > parameters.convergence_threshold:

        # Precompute variables
        u_with_bias = np.concatenate((mfmodel.u, np.ones((M, 1))), axis=1)
        v_with_bias = np.concatenate((mfmodel.v, np.expand_dims(mfmodel.b_n, axis=1)), axis=1)

        YTY = u_with_bias.T.dot(u_with_bias)
        lambdaI = np.eye(YTY.shape[0]) * mfmodel.lamb.lambda_v
        lambdaI[-1][-1] = lambdaI[-1][-1] * mfmodel.lamb.lambda_b_v / mfmodel.lamb.lambda_v
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
        lambdaI[-1][-1] = lambdaI[-1][-1] * mfmodel.lamb.lambda_b_u / mfmodel.lamb.lambda_u
        r = data - mfmodel.mu - mfmodel.b_n

        # Update latent variables
        for m in range(M):
            u_with_bias[m, :] = np.linalg.solve((XTX + lambdaI), r[m, :].dot(v_with_bias))

        # Extract variables and bias
        mfmodel.u = u_with_bias[:, :-1]
        mfmodel.b_m = u_with_bias[:, -1]

        predicted = mfmodel.calc_matrix()
        write_error_to_file(mfmodel, predicted, data, "ALS_train_error.txt")
        if extra_data_set is not None:
            write_error_to_file(mfmodel, predicted, extra_data_set, "ALS_test_error.txt")

        last_e = e
        e = mean_squared_error(mfmodel, predicted, data)
        print("Error: %f" % e)