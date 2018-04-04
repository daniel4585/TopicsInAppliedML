class Rating(object):
    def __init__(self, line):
        super(Rating, self)
        self.line = line

        splitted = line.split("::")
        self.user_id = int(splitted[0])
        self.movie_id = int(splitted[1])
        self.rating = int(splitted[2])
        self.timestamp = splitted[2]

    def __str__(self):
        return self.line

    def __repr__(self):
        return self.__str__()
