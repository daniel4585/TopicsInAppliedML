from DatasetSplit import DatasetSplit
from TrainTestSplit import TrainTestSplit
from ModelParameters import ModelParameters
from HyperParameters import HyperParameters
from SGD import SGD

def main():

    ds = DatasetSplit("data/datasetSplit.txt")
    print("Train size: " + str(len(ds.train)))
    print("Test size: " + str(len(ds.test)))
    trainTest = TrainTestSplit(ds, "data/datasetSentences.txt")
    modelParameters = ModelParameters(HyperParameters(20000, 50, 1, 50, 10, 100, 0.3, None, 211221, 0.3))
    modelParameters.Init(trainTest.train)

    sgd = SGD()
    sgd.LearnParamsUsingSGD(trainTest.train, modelParameters)


if __name__ == '__main__':
    main()
