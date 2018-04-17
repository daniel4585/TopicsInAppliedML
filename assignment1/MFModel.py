import numpy as np


class MFModel(object):
    def __init__(self,R,  K, lamb):
        self.K = K
        self.num_users, self.num_movies = R.shape

        # Initialize parameters randomly
        #todo initialize with small variance ~0.01
        self.u = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.v = np.random.normal(scale=1. / self.K, size=(self.num_movies, self.K))

        self.b_m = np.zeros(self.num_users)
        self.b_n = np.zeros(self.num_movies)

        # Initialize mu by taking the average of existing ratings
        self.mu = np.mean(R[np.where(R != 0)])

        self.lamb = lamb


    def calc_matrix(self):
        return (self.mu +
                    np.repeat(np.reshape(self.b_n, (self.num_movies, 1)), self.num_users, axis=1).T +
                        np.repeat(np.reshape(self.b_m, (self.num_users, 1)), self.num_movies, axis=1)) + self.u.dot(self.v.T)


