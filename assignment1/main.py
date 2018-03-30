from assignment1.train_test_split import train_test_split
from extractors import *
from Movie import Movie
from User import User
from Rating import Rating

def main():
    movies = extract_data("data/movies.dat", Movie)
    users = extract_data("data/users.dat", User)
    ratings = extract_data("data/ratings.dat", Rating)
    train, test = train_test_split(ratings)


if __name__ == '__main__':
    main()
