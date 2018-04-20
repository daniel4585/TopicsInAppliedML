from MFModel import MFModel
from GradientDecent import LearnModelFromDataUsingSGD, SGDParameters
from Lambda import Lambda
from AlternatingLeastSquares import LearnModelFromDataUsingALS, ALSParameters
from Evaluation import Evaluation
from Plots import *
from utils import *
from Movie import Movie
from User import User
from Rating import Rating
import pickle
import ConfigParser
import time
import os, errno

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

    bTrain = config.getboolean('Debug', 'bTrain')
    start = 0
    end = 0

    # Train model
    if bTrain:
        train, test = train_test_split(ratings)
        R_train = create_data_matrix(train, NUM_USERS, NUM_MOVIES)
        R_test = create_data_matrix(test, NUM_USERS, NUM_MOVIES)

        pickle_save('R_train.pkl', R_train)
        pickle_save('R_test.pkl', R_test)

        lamb = Lambda(config.getfloat('HyperParams', 'lambda_u'), config.getfloat('HyperParams', 'lambda_v'),
                      config.getfloat('HyperParams', 'lambda_b_u'),
                      config.getfloat('HyperParams', 'lambda_b_v'))

        model = MFModel(R_train, K=config.getint('Model', 'K'), lamb=lamb)

        start = time.time()

        if config.get('Model', 'chosen') == 'SGD':
            LearnModelFromDataUsingSGD(R_train, model, SGDParameters(steps=config.getint('SGD', 'steps'),
                                                                     alpha=config.getfloat('SGD', 'alpha')),
                                       extra_data_set=R_test)
            end = time.time()
            pickle_save('model_SGD.pkl', model)

        if config.get('Model', 'chosen') == 'ALS':
            LearnModelFromDataUsingALS(R_train, model,
                                       ALSParameters(convergence_threshold=int(config.get('ALS', 'threshold'))),
                                       extra_data_set=R_test)
            end = time.time()
            pickle_save('model_ALS.pkl', model)

        print "Time: " + str(end - start) + " s"

    # Don't train model - use saved pickle file
    if not bTrain:
        R_train = pickle_load('R_train.pkl')
        R_test = pickle_load('R_test.pkl')

        if config.get('Model', 'chosen') == 'SGD':
            model = pickle_load('model_SGD.pkl')
        if config.get('Model', 'chosen') == 'ALS':
            model = pickle_load('model_ALS.pkl')

    # Calculate metrics
    if config.getboolean('Debug', 'bMetrics'):
        e = Evaluation()
        e.calculate_ranks(model, R_test)
        rmse = str(e.rmse())
        print("RMSE: " + rmse)

        mpr = str(e.mpr())
        print("MPR: " + mpr)

        pA2 = str(e.patk(2))
        print("Average P@2: " + pA2)

        pA10 = str(e.patk(10))
        print("Average P@10: " + pA10)

        rA2 = str(e.ratk(2))
        print("Average R@2: " + rA2)

        rA10 = str(e.ratk(10))
        print("Average R@10: " + rA10)

        map = str(e.mean_average_precision())
        print("MAP: " + map)

        try:
            os.remove("output/results.txt")
        except OSError, IOError:
            pass

        with open("output/results.txt", 'a') as output:
            output.write("Hyperparameters: " + str(config.items('HyperParams')) + "\n")
            output.write("Chosen model: " + config.get('Model', 'chosen') + "\n")
            output.write("Metrics:" + "\n")
            output.write("\t" + "RMSE: " + rmse + "\n")
            output.write("\t" + "MPR: " + mpr + "\n")
            output.write("\t" + "Average P@2: " + pA2 + "\n")
            output.write("\t" + "Average P@10: " + pA10 + "\n")
            output.write("\t" + "Average R@2: " + rA2 + "\n")
            output.write("\t" + "Average R@10: " + rA10 + "\n")
            output.write("\t" + "MAP: " + map + "\n")
            output.write("Time: " + str(end - start) + " s" + "\n")
        plot_error(config.items('HyperParams'), config.items('SGD'))

    # Train models with different lambdas
    if config.getboolean('Debug', 'bLambda'):
        lamb_values = [0.1, 1, 10, 100]
        rmses = []
        mprs = []
        if config.getboolean('Debug', 'bLambdaTrain'):
            for value in lamb_values:
                print ("Training model with Lambda=" + str(value))
                lamb = Lambda(value, value, value, value)
                model = MFModel(R_train, K=config.getint('Model', 'K'), lamb=lamb)
                LearnModelFromDataUsingSGD(R_train, model, SGDParameters(steps=config.getint('SGD', 'steps'),
                                                                         alpha=config.getfloat('SGD', 'alpha')),
                                           extra_data_set=R_test)
                e = Evaluation()
                e.calculate_ranks(model, R_test)
                tmp = e.rmse()
                print("RMSE: " + str(tmp))
                rmses.append(tmp)

                tmp = e.mpr()
                mprs.append(tmp)
                print("MPR: " + str(tmp))
            pickle_save('rmses_lamb.pkl', rmses)
            pickle_save('mprs_lamb.pkl', mprs)
        else:
            rmses = pickle_load('rmses_lamb.pkl')
            mprs = pickle_load('mprs_lamb.pkl')
        plot_lambda(lamb_values, rmses, mprs, config.items('SGD'))

    # Train models with different Ks
    if config.getboolean('Debug', 'bK'):
        k_values = [2, 4, 10, 20, 40, 50, 70, 100, 200]

        rmses = []
        mprs = []
        times = []
        if config.getboolean('Debug', 'bKTrain'):
            for value in k_values:
                print ("Training model with K=" + str(value))
                lamb = Lambda(0.1, 0.1, 0.1, 0.1)
                model = MFModel(R_train, K=value, lamb=lamb)
                start = time.time()
                LearnModelFromDataUsingSGD(R_train, model, SGDParameters(steps=config.getint('SGD', 'steps'),
                                                                         alpha=config.getfloat('SGD', 'alpha')),
                                           extra_data_set=R_test)
                curr_time = time.time() - start
                print "Time: " + str(curr_time) + " s"
                times.append(curr_time)

                e = Evaluation()
                e.calculate_ranks(model, R_test)
                tmp = e.rmse()
                print("RMSE: " + str(tmp))
                rmses.append(tmp)

                tmp = e.mpr()
                mprs.append(tmp)
                print("MPR: " + str(tmp))
                pickle_save('rmses_k.pkl', rmses)
                pickle_save('mprs_k.pkl', mprs)
                pickle_save('times_k.pkl', mprs)
        else:
            rmses = pickle_load('rmses_k.pkl')
            mprs = pickle_load('mprs_k.pkl')
            times = pickle_load('times_k.pkl')

        plot_dim(k_values, rmses, mprs, config.items('SGD'))
        plot_dim_times(k_values,times, config.items('SGD'))

    # Summarize, print chosen users' recommendations
    e = Evaluation()
    e.calculate_ranks(model, R_test)
    users = [23, 99, 121, 123, 666]
    for user in users:
        e.print_user_recommendation_list(user, 3, movies_dict, users_dict)


if __name__ == '__main__':
    main()
