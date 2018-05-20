import pickle

from DatasetSplit import DatasetSplit
from TrainTestSplit import TrainTestSplit
from ModelParameters import ModelParameters
from HyperParameters import HyperParameters
from SGD import SGD


def pickle_load(path):
    with open(path, 'rb') as input:
        return pickle.load(input)


def pickle_save(path, value):
    with open(path, 'wb') as output:
        pickle.dump(value, output, pickle.HIGHEST_PROTOCOL)

def main():

    ds = DatasetSplit("data/datasetSplit.txt")
    print("Train size: " + str(len(ds.train)))
    print("Test size: " + str(len(ds.test)))
    trainTest = TrainTestSplit(ds, "data/datasetSentences.txt")
    modelParameters = ModelParameters(HyperParameters(100, 20, 5, 50, 20, 500, 0.3, None, 211221, 1.0, 100))
    modelParameters.Init(trainTest)

    if True:
        sgd = SGD()
        sgd.LearnParamsUsingSGD(trainTest, modelParameters)
        pickle_save('model.pkl', modelParameters)
    else:
        modelParameters = pickle_load('model.pkl')


if __name__ == '__main__':
    main()
