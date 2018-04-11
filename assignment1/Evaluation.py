import math
from collections import OrderedDict
from operator import itemgetter

import pickle


def calculate_ranks(mfmodel, test):
    bCalc = True


    predicted = mfmodel.calc_matrix()
    users_ranked_dicts = []
    users_ground_truth = []
    for user in range(mfmodel.num_users/20):
        users_ranked_dicts.append(ranked_dict_for_user(user, predicted))
        users_ground_truth.append(get_ground_truth(user, test))

    print("MPR: " + str(mpr(users_ground_truth, users_ranked_dicts)))
    print("RMSE: " + str(rmse(mfmodel, users_ground_truth, users_ranked_dicts)))


def get_ground_truth(n, data):
    indices = data[n, :].nonzero()[0]
    return zip(indices, data[n, indices])


def ranked_dict_for_user(n, data):
    templist = []
    for i, ranking in enumerate(data[n, :]):
        templist.append((i, ranking))

    movie_ranking = OrderedDict(sorted(templist, key=itemgetter(1), reverse=True))
    return movie_ranking


def get_predicted_rating(users_ranked_dicts, user, movie):
    return users_ranked_dicts[user][movie]

def get_movie_recomendation_index(users_ranked_dicts, user, movie):
    user_dict = users_ranked_dicts[user]
    index = 1
    for k, v in user_dict.items():
        if k == movie:
            return index
        index += 1
    raise Exception("didnt find movie")

def rmse(mfmodel, users_ground_truth, users_ranked_dicts):
    error = 0
    num_of_ratings = 0
    for user in range(mfmodel.num_users):
        user_error = 0
        num_of_ratings_per_user = len(users_ground_truth[user])
        num_of_ratings += num_of_ratings_per_user
        for movie, true_rating in users_ground_truth[user]:
            user_error = 0
            predicted_rating = get_predicted_rating(users_ranked_dicts, user, movie)

            user_error += (predicted_rating - true_rating) ** 2

        error += user_error
        #print("User: %d, Error: %f" % (user, math.sqrt(user_error / num_of_ratings_per_user)))

    return math.sqrt(error / num_of_ratings)


def mpr(users_ground_truth, users_ranked_dicts):
    sum_user_mpr = 0
    num_of_ratings = 0
    for user, ground_truth in enumerate(users_ground_truth):
        recomendation_index = 0
        for movie, _ in ground_truth:
            recomendation_index += get_movie_recomendation_index(users_ranked_dicts, user, movie)
        avg_recomendation_index = (1.*recomendation_index) / (1.*len(ground_truth))
        num_of_ratings += len(ground_truth)
        sum_user_mpr += (1.*avg_recomendation_index) / (1.*len(users_ranked_dicts[0]))
    return (1.*sum_user_mpr) / (1.*num_of_ratings)
