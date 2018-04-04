from MFModel import MFModel
from GradientDecent import LearnModelFromDataUsingSGD, Parameters
from Lambda import Lambda
from utils import *

from Movie import Movie
from User import User
from Rating import Rating

NUM_USERS = 6041  # One more than actual number of Users
NUM_MOVIES = 3953  # One more than actual number of Movies


def main():
    movies = extract_data("data/movies.dat", Movie)
    users = extract_data("data/users.dat", User)
    ratings = extract_data("data/ratings.dat", Rating)
    train, test = train_test_split(ratings)
    R = create_data_matrix(train, NUM_USERS, NUM_MOVIES)

    regularization = 0.2
    lamb = Lambda(lambda_u=regularization, lambda_v=regularization, lambda_b_u=regularization, lambda_b_v=regularization)
    model = MFModel(R, K=20, lamb=lamb)

    LearnModelFromDataUsingSGD(R, model, Parameters(steps=10, alpha=0.1))


if __name__ == '__main__':
    main()
