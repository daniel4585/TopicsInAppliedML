import math
from collections import OrderedDict
from operator import itemgetter

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

def calculate_tp(k, ranked_dict, user, users_ground_truth):
    TP = 0
    ground_truth = users_ground_truth[user]
    for movie in ranked_dict.keys()[:k]:
        for movie_truth, _ in ground_truth:
            if movie == movie_truth:
                TP += 1
                break
    return TP, ground_truth




class Evaluation(object):
    users_ranked_dicts = []
    users_ground_truth = []

    def calculate_ranks(self, mfmodel, test):
        predicted = mfmodel.calc_matrix()
        for user in range(mfmodel.num_users):
            self.users_ranked_dicts.append(ranked_dict_for_user(user, predicted))
            self.users_ground_truth.append(get_ground_truth(user, test))

    def print_user_recommendation_list(self, user, h, movies, users):
        user = user-1
        ground_truth = self.users_ground_truth[user]
        print "\n\nUser " + str(users[user])
        print("History:")
        for movie, rating in ground_truth:
             print "Movie: " + str(movies[movie]) + " Rating: " + str(rating)
        print "\nRecommended: "
        for i, kv_pair in enumerate(self.users_ranked_dicts[user].items()):
            print str(i) + ": " + str(movies[kv_pair[0]]) + " Rating: " + str(kv_pair[1])
            if i == h:
                break

    def rmse(self):
        error = 0
        num_of_ratings = 0
        for user in range(len(self.users_ranked_dicts)):
            user_error = 0
            num_of_ratings_per_user = len(self.users_ground_truth[user])
            num_of_ratings += num_of_ratings_per_user
            for movie, true_rating in self.users_ground_truth[user]:
                user_error = 0
                predicted_rating = get_predicted_rating(self.users_ranked_dicts, user, movie)

                user_error += (predicted_rating - true_rating) ** 2

            error += user_error
            # print("User: %d, Error: %f" % (user, math.sqrt(user_error / num_of_ratings_per_user)))

        return math.sqrt(error / num_of_ratings)

    def mpr(self):
        sum_user_mpr = 0
        num_of_ratings = 0
        for user, ground_truth in enumerate(self.users_ground_truth):
            recomendation_index = 0
            for movie, _ in ground_truth:
                recomendation_index += get_movie_recomendation_index(self.users_ranked_dicts, user, movie)
            avg_recomendation_index = (1. * recomendation_index) / (1. * len(ground_truth))
            num_of_ratings += len(ground_truth)
            sum_user_mpr += (1. * avg_recomendation_index) / (1. * len(self.users_ranked_dicts[0]))
        return (1. * sum_user_mpr) / (1. * num_of_ratings)

    def patk(self, k):
        totalTP = 0
        for user, ranked_dict in enumerate(self.users_ranked_dicts):
            TP, ground_truth = calculate_tp(k, ranked_dict, user, self.users_ground_truth)
            user_patk = (1. * TP) / (1. * k)
            # print("P@k = %f" % user_patk)
            totalTP += TP
        return (1. * totalTP) / (1. * k * len(self.users_ranked_dicts))

    def ratk(self, k):
        totalTP = 0
        totalGroundTruths = 0
        for user, ranked_dict in enumerate(self.users_ranked_dicts):
            TP, ground_truth = calculate_tp(k, ranked_dict, user, self.users_ground_truth)
            user_ratk = (1. * TP) / (1. * len(ground_truth))
            # print("R@k = %f" % user_ratk))
            totalGroundTruths += len(ground_truth)
            totalTP += TP

        return (1. * totalTP) / (1. * totalGroundTruths)

    def mean_average_precision(self):
        result = 0
        sum_ground_truths = 0
        for user, ranked_dict in enumerate(self.users_ranked_dicts):
            avg_precision = 0
            k = len(self.users_ground_truth[user])
            for k_iter in range(1, k + 1):
                # No need to calculate TP of last iteration
                user_ratk_1 = 0
                if k_iter != 1:
                    user_ratk_1 = (1. * TP) / (1. * len(ground_truth))

                TP, ground_truth = calculate_tp(k_iter, ranked_dict, user, self.users_ground_truth)
                user_ratp = (1. * TP) / (1. * k_iter)
                user_ratk = (1. * TP) / (1. * len(ground_truth))

                avg_precision += user_ratp * (user_ratk - user_ratk_1)

            # print("Average precision = %f" % avg_precision)
            result += avg_precision
            sum_ground_truths += len(ground_truth)
        return (1. * result ) / (1. * sum_ground_truths)
