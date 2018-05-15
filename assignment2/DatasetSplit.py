

class DatasetSplit(object):
    def __init__(self, filepath):
        super(DatasetSplit, self).__init__()
        self.train = set()
        self.test = set()

        with open(filepath, 'r') as f:
            for line in f.read().splitlines():
                splitted = line.split(",")
                if splitted[0].isdigit() and splitted[1].isdigit():
                    if splitted[1] == '1':
                        self.train.add(splitted[0])
                    if splitted[1] == '2':
                        self.test.add(splitted[0])

