import pickle
import time
import target_global

import numpy as np
import pandas as pd
from TrainData import TrainData
from ValidationData import ValidationData
from CART import CART, calculateLoss
from GBRT import GBRT
from Diagnostics import plot_TrainTestError
from Hyperparams import Hyperparams
import ConfigParser
from GBRT import GBRT_WithLoss
from Diagnostics import plot_varientParam
from Diagnostics import plot_Bar


def pickle_load(path):
    with open(path, 'rb') as input:
        return pickle.load(input)


def pickle_save(path, value):
    with open(path, 'wb') as output:
        pickle.dump(value, output, pickle.HIGHEST_PROTOCOL)





def main():
    config = ConfigParser.ConfigParser()
    config.read("conf.ini")


    target_global.target_name = config.get('Debug', 'target')

    df = pd.read_csv("data2/train.csv")
    df = df.drop("Id", axis=1)
    df = df[np.isfinite(df[target_global.target_name])]

    train, validation = np.split(df.sample(frac=1), [int(.8*len(df))])

    td = TrainData(train)
    vd = ValidationData(validation, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

    if config.getboolean('Debug', 'deliv2'):

        hyperParams1 = Hyperparams(maxDepth=16, eta=0.5, nu=1.0, numOfTrees=100, minNodeSize=5)
        ensemble = GBRT(td.df, vd.df, hyperparams=hyperParams1, outputFile="results_1.txt")
        pickle_save('ensemble_1_data2.pkl', ensemble)

        hyperParams2 = Hyperparams(maxDepth=64, eta=0.5, nu=1.0, numOfTrees=50, minNodeSize=5)
        ensemble = GBRT(td.df,vd.df, hyperparams=hyperParams2, outputFile="results_2.txt")
        pickle_save('ensemble_2_data2.pkl', ensemble)

        hyperParams3 = Hyperparams(maxDepth=16, eta=0.5, nu=1.0, numThresholds=10, numOfTrees=50, minNodeSize=5)
        ensemble = GBRT(td.df, vd.df, hyperparams=hyperParams3, outputFile="results_3.txt")
        pickle_save('ensemble_3_data2.pkl', ensemble)

        hyperParams4 = Hyperparams(maxDepth=16, eta=0.75, nu=1.0, numOfTrees=50, minNodeSize=5)
        ensemble = GBRT(td.df, vd.df, hyperparams=hyperParams4, outputFile="results_4.txt")
        pickle_save('ensemble_4_data2.pkl', ensemble)

        plot_TrainTestError(hyperParams1, "results_1.txt")
        plot_TrainTestError(hyperParams2, "results_2.txt")
        plot_TrainTestError(hyperParams3, "results_3.txt")
        plot_TrainTestError(hyperParams4, "results_4.txt")

    if config.getboolean('Debug', 'deliv3.1'):
        maxDepthValues = [2**n for n in range(1, 8)]
        maxDepthTimes=[]
        finalTrainLoss = []
        finalTestLoss = []

        # maxDepthTimes = pickle_load("max_depth_times.pkl")
        # finalTrainLoss = pickle_load("max_depth_trainloss.pkl")
        # finalTestLoss = pickle_load("max_depth_testloss.pkl")
        hyperParams = Hyperparams(maxDepth=2, eta=0.75, nu=1.0, numOfTrees=50, minNodeSize=5)
        plot_varientParam(hyperParams, maxDepthValues, finalTrainLoss, finalTestLoss, maxDepthTimes, "Max Tree Depth",
                          "Max Tree Depth")

        for maxDepth in maxDepthValues:
            start = time.time()
            hyperParams = Hyperparams(maxDepth=maxDepth, eta=0.75, nu=1.0, numOfTrees=50, minNodeSize=5)
            ensemble, trainLoss, testLoss = GBRT_WithLoss(td.df, vd.df, hyperparams=hyperParams)
            maxDepthTimes.append(time.time() - start)
            finalTrainLoss.append(trainLoss)
            finalTestLoss.append(testLoss)
            pickle_save("d2_max_depth_times.pkl", maxDepthTimes)
            pickle_save("d2_max_depth_trainloss.pkl", finalTrainLoss)
            pickle_save("d2_max_depth_testloss.pkl", finalTestLoss)
        plot_varientParam(hyperParams, maxDepthValues, finalTrainLoss, finalTestLoss, maxDepthTimes, "Max Tree Depth",
                          "Max Tree Depth")


    if config.getboolean('Debug', 'deliv3.2'):
        numOfThresholdsValues = [2**n for n in range(2, 9)]
        numOfThresholdsTimes=[]
        finalTrainLoss = []
        hyperParams = Hyperparams(maxDepth=16, eta=0.75, nu=1.0, numOfTrees=75, minNodeSize=5)
        finalTestLoss = []
        for numOfThresholds in numOfThresholdsValues:
            start = time.time()
            ensemble, trainLoss, testLoss = GBRT_WithLoss(td.df, vd.df, hyperparams=hyperParams)
            numOfThresholdsTimes.append(time.time() - start)
            finalTrainLoss.append(trainLoss)
            finalTestLoss.append(testLoss)
            pickle_save("d2_numOfThresholds_times.pkl", numOfThresholdsTimes)
            pickle_save("d2_numOfThresholds_trainloss.pkl", finalTrainLoss)
            pickle_save("d2_numOfThresholds_testloss.pkl", finalTestLoss)



        numOfThresholdsTimes = pickle_load("d2_numOfThresholds_times.pkl")
        finalTrainLoss = pickle_load("d2_numOfThresholds_trainloss.pkl")
        finalTestLoss = pickle_load("d2_numOfThresholds_testloss.pkl")
        plot_varientParam(hyperParams, numOfThresholdsValues, finalTrainLoss, finalTestLoss, numOfThresholdsTimes,
                          "Num of Thresholds", "Num of Thresholds")


    if config.getboolean('Debug', 'deliv4'):
        ensemble = pickle_load("ensemble_2_data2.pkl")
        for i in range(5):
            print ensemble.trees[i]
        plot_Bar(ensemble.getFeatureImprortance(td.df))


    if config.getboolean('Debug', 'train'):
        hyperParams = Hyperparams(maxDepth=16, eta=0.5, nu=1.0, numThresholds=10, numOfTrees=77, minNodeSize=5)

        #8 - 1394613139

        ensemble = GBRT(td.df, vd.df, hyperparams=hyperParams, outputFile="final_model.txt")
        pickle_save('final_model.pkl', ensemble)
        ensemble = pickle_load('final_model.pkl')

        for i in range(5):
            print ensemble.trees[i]
        plot_TrainTestError(hyperParams, "final_model.txt")


    if config.getboolean('Debug', 'test'):

        ensemble = pickle_load('final_model.pkl')
        test = pd.read_csv("data/test.csv")
        testData = ValidationData(test, td.cat_mapping, td.cat_mapping_avg, td.numerical_mapping)

        with open("output/submission.csv", "w") as prediction:
            prediction.write("Id,SalePrice\n")

            for index, row in testData.df.iterrows():
                prediction.write(str(int(row["Id"])) + "," + str(ensemble.Evaluate(row)) + "\n")


if __name__ == '__main__':
    main()
