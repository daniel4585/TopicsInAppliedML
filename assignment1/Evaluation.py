import math


def calculate_ranks(mfmodel, test):
    predicted = mfmodel.calc_matrix()
    users_ranked_lists = []
    users_ground_truth = []
    for user in range(mfmodel.num_users):
        users_ranked_lists.append(ranked_list_for_user(user, predicted))
        users_ground_truth.append(get_ground_truth(user, test))

    print("RMSE: " + str(rmse(users_ground_truth, users_ranked_lists)))




def get_ground_truth(n, data):
    indices = data[n, :].non_zero()[0]
    return indices, data[n, indices]


def ranked_list_for_user(n, data):
    movie_ranking = []
    for i, ranking in enumerate(data[n, :]):
        movie_ranking.append((i, ranking))

    return movie_ranking


def get_predicted_rating(users_ranked_lists, user, movie):
    for predicted_movie in users_ranked_lists[user]:
        if predicted_movie[0] == movie:
            return predicted_movie[1]


def rmse(users_ground_truth, users_ranked_lists):
    error = 0
    num_of_ratings = 0
    for user, ground_truth in users_ground_truth:
        user_error = 0
        num_of_ratings_per_user = 0
        for movie in ground_truth[0]:
            predicted_rating = get_predicted_rating(users_ranked_lists, user, movie)
            rating = ground_truth[1][movie]

            user_error += (predicted_rating - rating) ** 2
            num_of_ratings_per_user += 1

        error += user_error
        num_of_ratings += num_of_ratings_per_user
        print("User: %d, Error: %f" % (user, math.sqrt(user_error / num_of_ratings_per_user)))

    return math.sqrt(error / num_of_ratings)
