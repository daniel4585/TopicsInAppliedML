from assignment1.train_test_split import *
from extractors import *
from Movie import Movie
from User import User
from Rating import Rating

NUM_USERS = 6041 # One more than actual number of Users
NUM_MOVIES = 3953 # One more than actual number of Movies

def main():
    movies = extract_data("data/movies.dat", Movie)
    users = extract_data("data/users.dat", User)
    ratings = extract_data("data/ratings.dat", Rating)
    train, test = train_test_split(ratings)
    R = create_data_matrix(train, NUM_USERS, NUM_MOVIES)

    print R[2]
    print "---"



if __name__ == '__main__':
    main()
