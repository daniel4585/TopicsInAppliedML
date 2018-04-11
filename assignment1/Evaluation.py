import math
from collections import OrderedDict
from operator import itemgetter


def calculate_ranks(mfmodel, test):

    predicted = mfmodel.calc_matrix()
    users_ranked_dicts = []
    users_ground_truth = []
    for user in range(mfmodel.num_users/20):
        users_ranked_dicts.append(ranked_dict_for_user(user, predicted))
        users_ground_truth.append(get_ground_truth(user, test))

    print("RMSE: " + str(rmse(users_ground_truth, users_ranked_dicts)))
    print("MPR: " + str(mpr(users_ground_truth, users_ranked_dicts)))

    k = 10
    print("Average P@k: " + str(patk(users_ground_truth, users_ranked_dicts, k)))
    print("Average R@k: " + str(ratk(users_ground_truth, users_ranked_dicts, k)))


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


def rmse(users_ground_truth, users_ranked_dicts):
    error = 0
    num_of_ratings = 0
    for user in range(len(users_ranked_dicts.keys())):
        user_error = 0
        num_of_ratings_per_user = len(users_ground_truth[user])
        num_of_ratings += num_of_ratings_per_user
        for movie, true_rating in users_ground_truth[user]:
            user_error = 0
            predicted_rating = get_predicted_rating(users_ranked_dicts, user, movie)

            user_error += (predicted_rating - true_rating) ** 2

        error += user_error
        # print("User: %d, Error: %f" % (user, math.sqrt(user_error / num_of_ratings_per_user)))

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


def patk(users_ground_truth, users_ranked_dicts, k):
    totalTP = 0
    for user, ranked_dict in enumerate(users_ranked_dicts):
        TP, ground_truth = calculate_tp(k, ranked_dict, user, users_ground_truth)
        # print("P@k = %f" % ((1. * TP) / (1. * k)))
        totalTP += TP

    return (1. * totalTP) / (1. * k * len(users_ranked_dicts))


def ratk(users_ground_truth, users_ranked_dicts, k):
    totalTP = 0
    totalGroundTruths = 0
    for user, ranked_dict in enumerate(users_ranked_dicts):
        TP, ground_truth = calculate_tp(k, ranked_dict, user, users_ground_truth)
        totalGroundTruths += len(ground_truth)
        totalTP += TP
        # print("R@k = %f" % ((1. * TP) / (1. * len(ground_truth))))

    return (1. * totalTP) / (1. * totalGroundTruths)


def calculate_tp(k, ranked_dict, user, users_ground_truth):
    TP = 0
    ground_truth = users_ground_truth[user]
    for movie in ranked_dict.keys()[:k]:
        for movie_truth, _ in ground_truth:
            if movie == movie_truth:
                TP += 1
                break

    return TP, ground_truth
