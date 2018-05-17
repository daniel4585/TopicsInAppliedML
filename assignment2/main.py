from DatasetSplit import DatasetSplit
from TrainTestSplit import TrainTestSplit
from ModelParameters import ModelParameters
from HyperParameters import HyperParameters




def main():

    ds = DatasetSplit("data/datasetSplit.txt")
    print("Train size: " + str(len(ds.train)))
    print("Test size: " + str(len(ds.test)))
    trainTest = TrainTestSplit(ds, "data/datasetSentences.txt")
    modelParameters = ModelParameters(HyperParameters(1, 10, 100, 20, None, 211221,0.75))
    modelParameters.Init(trainTest.train)
    wt, wc = modelParameters.sample_target_context(trainTest.train)
    print wt
    print wc
    wt, wc = modelParameters.sample_target_context(trainTest.train)
    print wt
    print wc



if __name__ == '__main__':
    main()
