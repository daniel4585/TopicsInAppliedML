from MFModel import MFModel
from GradientDecent import LearnModelFromDataUsingSGD, SGDParameters
from Lambda import Lambda
from AlternatingLeastSquares import LearnModelFromDataUsingALS, ALSParameters
from Evaluation import Evaluation
from utils import *
from Movie import Movie
from User import User
from Rating import Rating
import pickle
import ConfigParser



NUM_USERS = 6040
NUM_MOVIES = 3952


def main():
    movies = extract_data("data/movies.dat", Movie)
    movies_dict = dict(zip([x.id for x in movies], movies))
    users = extract_data("data/users.dat", User)
    users_dict = dict(zip([x.id for x in users], users))
    ratings = extract_data("data/ratings.dat", Rating)

    config = ConfigParser.ConfigParser()
    config.read("conf.ini")

    bTrain = config.get('Model', 'bTrain') == 'True'


    if bTrain:
        train, test = train_test_split(ratings)
        R_train = create_data_matrix(train, NUM_USERS, NUM_MOVIES)
        R_test = create_data_matrix(test, NUM_USERS, NUM_MOVIES)

        with open('R_train.pkl', 'wb') as output:
            pickle.dump(R_train, output, pickle.HIGHEST_PROTOCOL)

        with open('R_test.pkl', 'wb') as output:
            pickle.dump(R_test, output, pickle.HIGHEST_PROTOCOL)

            lamb = Lambda(float(config.get('HyperParams', 'lambda_u')), float(config.get('HyperParams', 'lambda_v')),
                          float(config.get('HyperParams', 'lambda_b_u')),
                          float(config.get('HyperParams', 'lambda_b_v')))

            model = MFModel(R_train, K=int(config.get('Model', 'K')), lamb=lamb)

            if config.get('Model', 'chosen') == 'SGD':
                LearnModelFromDataUsingSGD(R_train, model, SGDParameters(steps=float(config.get('SGD', 'steps')),
                                                                         alpha=float(config.get('SGD', 'alpha'))))
                with open('model_SGD.pkl', 'wb') as output:
                    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

            if config.get('Model', 'chosen') == 'ALS':
                LearnModelFromDataUsingALS(R_train, model,
                                           ALSParameters(convergence_threshold=int(config.get('ALS', 'threshold'))))

                with open('model_ALS.pkl', 'wb') as output:
                    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    if not bTrain:
        with open('R_train.pkl', 'rb') as input:
            R_train = pickle.load(input)
        with open('R_test.pkl', 'rb') as input:
            R_test = pickle.load(input)

        if config.get('Model', 'chosen') == 'SGD':
            with open('model_SGD.pkl', 'rb') as input:
                model = pickle.load(input)
        if config.get('Model', 'chosen') == 'ALS':
            with open('model_ALS.pkl', 'rb') as input:
                model = pickle.load(input)


    e = Evaluation()
    e.calculate_ranks(model, R_train)
    print("RMSE: " + str(e.rmse()))
    print("MPR: " + str(e.mpr()))

    k = 20
    print("Average P@k: " + str(e.patk(k)))
    print("Average R@k: " + str(e.ratk(k)))
    print("MAP: " + str(e.mean_average_precision(k)))

    e.print_user_recommendation_list(5, 5, movies_dict, users_dict)
    e.print_user_recommendation_list(22, 7, movies_dict, users_dict)
    e.print_user_recommendation_list(55, 5, movies_dict, users_dict)
    e.print_user_recommendation_list(88, 5, movies_dict, users_dict)


if __name__ == '__main__':
    main()
