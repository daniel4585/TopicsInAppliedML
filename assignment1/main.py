from MFModel import MFModel
from GradientDecent import LearnModelFromDataUsingSGD, SGDParameters
from Lambda import Lambda
from AlternatingLeastSquares import LearnModelFromDataUsingALS, ALSParameters
from utils import *
from Movie import Movie
from User import User
from Rating import Rating
import pickle


NUM_USERS = 6040
NUM_MOVIES = 3952


def main():
    # movies = extract_data("data/movies.dat", Movie)
    # users = extract_data("data/users.dat", User)
    # ratings = extract_data("data/ratings.dat", Rating)
    # train, test = train_test_split(ratings)
    # R = create_data_matrix(train, NUM_USERS, NUM_MOVIES)
    #
    # with open('R.pkl', 'wb') as output:
    #      pickle.dump(R, output, pickle.HIGHEST_PROTOCOL)

    with open('R.pkl', 'rb') as input:
        R = pickle.load(input)

    # regularization = 0.1
    # lamb = Lambda(lambda_u=regularization, lambda_v=regularization, lambda_b_u=regularization,
    #               lambda_b_v=regularization)
    # model = MFModel(R, K=20, lamb=lamb)
    # LearnModelFromDataUsingSGD(R, model, SGDParameters(steps=10, alpha=0.01))

    regularization = 0.1
    lamb = Lambda(lambda_u=regularization, lambda_v=regularization, lambda_b_u=regularization,
                  lambda_b_v=regularization)
    model = MFModel(R, K=20, lamb=lamb)
    LearnModelFromDataUsingALS(R, model, ALSParameters(convergence_threshold=30))


if __name__ == '__main__':
    main()
