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
