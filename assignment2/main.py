from DatasetSplit import DatasetSplit


def main():

    ds = DatasetSplit("data/datasetSplit.txt")
    print("Train size: " + str(len(ds.train)))
    print("Test size: " + str(len(ds.test)))


if __name__ == '__main__':
    main()
