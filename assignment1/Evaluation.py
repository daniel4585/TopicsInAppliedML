import math
from collections import OrderedDict
from operator import itemgetter


def calculate_ranks(mfmodel, test):
    predicted = mfmodel.calc_matrix()
    users_ranked_dicts = []
    users_ground_truth = []
    for user in range(mfmodel.num_users):
        users_ranked_dicts.append(ranked_dict_for_user(user, predicted))
        users_ground_truth.append(get_ground_truth(user, test))

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
    for predicted_movie in users_ranked_dicts[user]:
        return predicted_movie[movie]

def get_movie_recomendation_index(users_ranked_lists, user, movie):


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


def mpr(users_ground_truth, users_ranked_lists):
    sum_user_mpr = 0
    num_of_ratings = 0
    for user, ground_truth in users_ground_truth:
        num_of_ratings_per_user = 0
        recomendation_index = 0
        for movie in ground_truth[0]:
            recomendation_index += get_movie_recomendation_index(users_ranked_lists, user, movie)
            num_of_ratings_per_user += 1
        avg_recomendation_index = recomendation_index / num_of_ratings_per_user
        num_of_ratings += num_of_ratings_per_user
        sum_user_mpr += avg_recomendation_index / len(users_ranked_lists[0])
    return sum_user_mpr /  num_of_ratings