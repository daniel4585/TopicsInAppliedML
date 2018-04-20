import random
import numpy as np
import pickle


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

    print "Train size: " + str(len(train))
    print "Test size: " + str(len(test))
    #print "together " + str(len(train) + len(test))
    #print "rating list " + str(len(rating_list))
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

def mean_squared_error(mfmodel, predicted, data):
    xs, ys = data.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        error += pow(data[x, y] - predicted[x, y], 2) / 2

    for i in range(mfmodel.num_movies):
        error += mfmodel.lamb.lambda_v * np.linalg.norm(mfmodel.v[i, :], ord=2) ** 2 / 2

    for i in range(mfmodel.num_users):
        error += mfmodel.lamb.lambda_u * np.linalg.norm(mfmodel.u[i, :], ord=2) ** 2 / 2

    error += mfmodel.lamb.lambda_b_u * (mfmodel.b_m**2).sum() / 2
    error += mfmodel.lamb.lambda_b_v * (mfmodel.b_n**2).sum() / 2

    return error


def write_error_to_file(mfmodel, predicted, data, file_output):
    error = mean_squared_error(mfmodel, predicted, data)
    with open("output/" + file_output, 'a') as output:
         output.write("error:" + str(error) + "\n")


def pickle_load(path):
    with open(path, 'rb') as input:
        return pickle.load(input)


def pickle_save(path, value):
    with open(path, 'wb') as output:
        pickle.dump(value, output, pickle.HIGHEST_PROTOCOL)
