class User(object):
    def __init__(self, line):
        super(User, self)
        self.line = line

        splitted = line.split("::")
        self.id = int(splitted[0]) - 1
        self.gender = splitted[1]
        self.age = splitted[2]
        self.occupation = splitted[3]
        self.zipcode = splitted[4]

    def __str__(self):
        return "id: " + str(self.id) + " gender: " + str(self.gender) + " age: " + str(self.age) + " occupation " + (
            self.occupation) + " zip " + str(self.zipcode)
