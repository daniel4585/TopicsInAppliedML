import string
from DatasetSplit import DatasetSplit

class TrainTestSplit(object):
    def __init__(self, dsSplit, sentancesPath):
        self.test = []
        self.train = []
        with open(sentancesPath) as f:
            content = f.read().splitlines(False)
        for line in content:
            senId = line.split("\t")[0]
            if not senId.isdigit():
                continue
            sentance = line.split("\t")[1]
            tokenized = self.tokenize_line(sentance)
            if senId in dsSplit.test:
                self.test.append(tokenized)
            if senId in dsSplit.train:
                self.train.append(tokenized)


    def tokenize_line(self, sentance):
        ascii = set(string.ascii_letters)
        words = [x for x in sentance.split(" ")]
        alphaNumeric = [''.join(c.lower() for c in word if c.isalpha() and c in ascii) for word in words]
        ret = [x for x in alphaNumeric if len(x) > 2]
        return ret












