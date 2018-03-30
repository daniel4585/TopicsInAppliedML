

from extractors import *


def main():
    movies_arr = movies_extract("data/movies.dat")
    for mov in movies_arr:
        print mov


if __name__ == '__main__':
    main()