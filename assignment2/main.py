from DatasetSplit import DatasetSplit
from TrainTestSplit import TrainTestSplit

def main():

    ds = DatasetSplit("data/datasetSplit.txt")
    print("Train size: " + str(len(ds.train)))
    print("Test size: " + str(len(ds.test)))
    trainTest = TrainTestSplit(ds, "data/datasetSentences.txt")



if __name__ == '__main__':
    main()
