import random


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
            train.extend(user_ratings[:int(len(user_ratings)*0.8)])
            test.extend(user_ratings[int(len(user_ratings)*0.8):])
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


