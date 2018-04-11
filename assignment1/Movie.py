class Movie(object):
    def __init__(self, line):
        super(Movie, self)
        self.line = line

        splitted = line.split("::")
        self.id = int(splitted[0]) - 1
        self.title = splitted[1]
        self.genres = splitted[2]

    def __str__(self):
        return str(self.id + 1) + ": " + self.title + " :: " + self.genres
