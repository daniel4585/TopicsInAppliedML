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
    modelParameters = ModelParameters(HyperParameters(20000, 50, 5, 50, 10, 10000, 0.3, None, 211221, 1.0, 100))
    modelParameters.Init(trainTest)

    sgd = SGD()
    sgd.LearnParamsUsingSGD(trainTest, modelParameters)


if __name__ == '__main__':
    main()
