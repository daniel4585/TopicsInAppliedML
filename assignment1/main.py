from MFModel import MFModel
from GradientDecent import LearnModelFromDataUsingSGD, SGDParameters
from Lambda import Lambda
from AlternatingLeastSquares import LearnModelFromDataUsingALS, ALSParameters
from Evaluation import calculate_ranks, Evaluation
from utils import *
from Movie import Movie
from User import User
from Rating import Rating
import pickle


NUM_USERS = 6040
NUM_MOVIES = 3952


def main():
    bTrain = False
    if bTrain:
        movies = extract_data("data/movies.dat", Movie)
        users = extract_data("data/users.dat", User)
        ratings = extract_data("data/ratings.dat", Rating)
        train, test = train_test_split(ratings)
        R_train = create_data_matrix(train, NUM_USERS, NUM_MOVIES)
        R_test = create_data_matrix(test, NUM_USERS, NUM_MOVIES)

        with open('R_train.pkl', 'wb') as output:
            pickle.dump(R_train, output, pickle.HIGHEST_PROTOCOL)
    if not bTrain:
        with open('R_train.pkl', 'rb') as input:
            R_train = pickle.load(input)

    if bTrain:
        with open('R_test.pkl', 'wb') as output:
            pickle.dump(R_test, output, pickle.HIGHEST_PROTOCOL)

    if not bTrain:
        with open('R_test.pkl', 'rb') as input:
            R_test = pickle.load(input)

    if bTrain:
        regularization = 0.1
        lamb = Lambda(lambda_u=regularization, lambda_v=regularization, lambda_b_u=regularization,
                      lambda_b_v=regularization)
        model = MFModel(R_train, K=30, lamb=lamb)
        LearnModelFromDataUsingSGD(R_train, model, SGDParameters(steps=10, alpha=0.01))

        regularization = 0.03
        lamb = Lambda(lambda_u=regularization, lambda_v=regularization, lambda_b_u=regularization,
                      lambda_b_v=regularization)
        model = MFModel(R_train, K=30, lamb=lamb)
        LearnModelFromDataUsingALS(R_train, model, ALSParameters(convergence_threshold=3000))

        with open('model_ALS.pkl', 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    if not bTrain:
        with open('model_ALS.pkl', 'rb') as input:
            model = pickle.load(input)

    e = Evaluation()
    e.calculate_ranks(model, R_test)


if __name__ == '__main__':
    main()
