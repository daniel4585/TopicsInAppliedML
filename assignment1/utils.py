import random
import numpy as np


def train_test_split(rating_list):
    train = []
    test = []
    user_ratings = []
    curr_id = rating_list[0].user_id
    for rating in rating_list:
        if curr_id == rating.user_id:
            user_ratings.append(rating)
        else:
            random.shuffle(user_ratings)
            train.extend(user_ratings[:int(len(user_ratings) * 0.8)])
            test.extend(user_ratings[int(len(user_ratings) * 0.8):])
            curr_id = rating.user_id
            user_ratings = [rating]
    random.shuffle(user_ratings)
    train.extend(user_ratings[:int(len(user_ratings) * 0.8)])
    test.extend(user_ratings[int(len(user_ratings) * 0.8):])

    print "train " + str(len(train))
    print "test " + str(len(test))
    print "together " + str(len(train) + len(test))
    print "rating list " + str(len(rating_list))
    return train, test


def create_data_matrix(data, num_users, num_movies):
    R = np.zeros(shape=(num_users, num_movies))
    for rating in data:
        R[rating.user_id, rating.movie_id] = rating.rating
    return R


def extract_data(path, clazz):
    with open(path) as f:
        content = f.readlines()

    # Remove whitespaces and \n
    content = [x.strip() for x in content]

    data = []
    for line in content:
        data.append(clazz(line))

    return data
