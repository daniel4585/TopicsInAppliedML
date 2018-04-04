import numpy as np


class MFModel(object):
    def __init__(self, R, K, lamb):
        self.K = K
        self.R = R
        self.num_users, self.num_movies = R.shape

        # Initialize parameters randomly
        self.u = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.v = np.random.normal(scale=1. / self.K, size=(self.num_movies, self.K))

        self.b_m = np.zeros(self.num_users)
        self.b_n = np.zeros(self.num_movies)

        # Initialize mu by taking the average of existing ratings
        self.mu = np.mean(R[np.where(R != 0)])

        self.lamb = lamb

    def calc_matrix(self):
        return (self.mu + self.b_n[:, np.newaxis] + self.b_m[np.newaxis, :]).T + self.u.dot(self.v.T)

    def mean_squared_error(self, predicted):
        xs, ys = self.R.nonzero()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2) / 2

        for i in range(self.num_movies):
            error += self.lamb.lambda_v * np.linalg.norm(self.v[i, :], ord=2) ** 2 / 2

        for i in range(self.num_users):
            error += self.lamb.lambda_u * np.linalg.norm(self.u[i, :], ord=2) ** 2 / 2

        error += self.lamb.lambda_b_u * (self.b_m**2).sum() / 2
        error += self.lamb.lambda_b_u * (self.b_n**2).sum() / 2
        return error
