from Movie import Movie

def movies_extract(path):

    with open(path) as f:
        content = f.readlines()

    # Remove whitespaces and \n
    content = [x.strip() for x in content]

    movies = []
    for line in content:
        movies.append(Movie(line))

    return movies
