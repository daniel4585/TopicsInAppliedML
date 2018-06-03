import pickle
from Plots import *

from DatasetSplit import DatasetSplit
from TrainTestSplit import TrainTestSplit
from ModelParameters import ModelParameters
from HyperParameters import HyperParameters
from SGD import SGD
from Evaluation import *
import ConfigParser
import time

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
    config = ConfigParser.ConfigParser()
    config.read("conf.ini")

    modelParameters = ModelParameters(HyperParameters(config.getint('HyperParams', 'iterations'), config.getint('HyperParams', 'minibatchsize'),
                                                       config.getint('HyperParams', 'C'), config.getint('HyperParams', 'D'),
                                                       config.getint('HyperParams', 'K'), config.getint('HyperParams', 'annealingRate'), config.getfloat('HyperParams', 'eta'),
                                                       config.getint('HyperParams', 'seed'), config.getfloat('HyperParams', 'alpha'),config.getint('HyperParams', 'X')))
    # modelParameters.Init(trainTest)

    if config.getboolean('Debug', 'bTrain'):
        start = time.time()
        sgd = SGD()
        finalTestLogLiklihood = sgd.LearnParamsUsingSGD(trainTest, modelParameters)
        total_time = time.time()-start
        pickle_save('model.pkl', modelParameters)

        with open("output/results.txt", 'a') as output:
            output.write("Hyperparameters: " + str(config.items('HyperParams')) + "\n")
            output.write("Train time: " + str(total_time/3600) + "H " + str(total_time % 3600 / 60) + "M" + str(total_time % 3600 % 60) + "S" + "\n")
            output.write("Final Test Logliklihood: " + str(finalTestLogLiklihood))

    else:
       #modelParameters = pickle_load('model.pkl')
        modelParameters = pickle_load('eta_2.0_model.pkl')


    #plot_logLiklihood(modelParameters.hyperParams)

    if config.getboolean('Debug', 'bVariantD'):
        total_time = []
        finalTestLogLiklihood = []
        finalTrainLogLiklihood = []
        d_vals = np.linspace(10.0, 300.0, 5, dtype=int)
        # for d in d_vals:
        #     start = time.time()
        #     modelParameters = ModelParameters(HyperParameters(config.getint('HyperParams', 'iterations'),
        #                                                       config.getint('HyperParams', 'minibatchsize'),
        #                                                       config.getint('HyperParams', 'C'),
        #                                                       d,
        #                                                       config.getint('HyperParams', 'K'),
        #                                                       config.getint('HyperParams', 'annealingRate'),
        #                                                       config.getfloat('HyperParams', 'eta'),
        #                                                       config.getint('HyperParams', 'seed'),
        #                                                       config.getfloat('HyperParams', 'alpha'),
        #                                                       config.getint('HyperParams', 'X')))
        #     modelParameters.Init(trainTest)
        #     sgd = SGD()
        #     finalTestLogLiklihood.append(sgd.LearnParamsUsingSGD(trainTest, modelParameters))
        #     finalTrainLogLiklihood.append(loglikelihood(trainTest.train, modelParameters))
        #     total_time.append(time.time() - start)
        #     pickle_save(str(d) + "_finalTrainLogLiklihood.pkl", finalTrainLogLiklihood)
        #     pickle_save(str(d) + "_finalTestLogLiklihood.pkl", finalTestLogLiklihood)
        #     pickle_save(str(d) + "_d_total_time.pkl", total_time)

        #pickle_save("d_finalTrainLogLiklihood.pkl", finalTrainLogLiklihood)
        #pickle_save("d_finalTestLogLiklihood.pkl", finalTestLogLiklihood)
        #pickle_save("d_total_time.pkl", total_time)
        finalTrainLogLiklihood = pickle_load("d_finalTrainLogLiklihood.pkl")
        finalTestLogLiklihood = pickle_load("d_finalTestLogLiklihood.pkl")
        total_time = pickle_load("d_total_time.pkl")
        plot_varientParam(modelParameters.hyperParams, d_vals, finalTrainLogLiklihood, finalTestLogLiklihood, total_time, "size of the word embedding-D. ", 'size of the word embedding - D')


    if config.getboolean('Debug', 'bVariantLearnRate'):
        total_time = []
        finalTestLogLiklihood = []
        finalTrainLogLiklihood = []
        sgd = SGD()
        eta_vals = np.linspace(0.1, 2, 5)
        # for eta in eta_vals:
        #     start = time.time()
        #     modelParameters = ModelParameters(HyperParameters(config.getint('HyperParams', 'iterations'),
        #                                                       config.getint('HyperParams', 'minibatchsize'),
        #                                                       config.getint('HyperParams', 'C'),
        #                                                       config.getint('HyperParams', 'D'),
        #                                                       config.getint('HyperParams', 'K'),
        #                                                       config.getint('HyperParams', 'annealingRate'),
        #                                                       eta,
        #                                                       config.getint('HyperParams', 'seed'),
        #                                                       config.getfloat('HyperParams', 'alpha'),
        #                                                       config.getint('HyperParams', 'X')))
        #     modelParameters.Init(trainTest)
        #     finalTestLogLiklihood.append(sgd.LearnParamsUsingSGD(trainTest, modelParameters))
        #     finalTrainLogLiklihood.append(loglikelihood(trainTest.train, modelParameters))
        #     total_time.append(time.time() - start)
        #     pickle_save("eta_" + str(eta) + "_model.pkl", modelParameters)

        #pickle_save("eta_finalTrainLogLiklihood.pkl", finalTrainLogLiklihood)
        #pickle_save("eta_finalTestLogLiklihood.pkl", finalTestLogLiklihood)
        #pickle_save("eta_total_time.pkl", total_time)
        finalTrainLogLiklihood = pickle_load("eta_finalTrainLogLiklihood.pkl")
        finalTestLogLiklihood = pickle_load("eta_finalTestLogLiklihood.pkl")
        total_time = pickle_load("eta_total_time.pkl")
        plot_varientParam(modelParameters.hyperParams, eta_vals, finalTrainLogLiklihood, finalTestLogLiklihood, total_time, "eta", "eta")

    if config.getboolean('Debug', 'bEvaluate'):
        print "Best Context words:"
        for word in ["good", "bad", "lame", "cool", "exciting"]:
            print "\t Target:" + word
            print "\t " + str(PredictContext(modelParameters, word))

        modelParameters = ModelParameters(HyperParameters(config.getint('HyperParams', 'iterations'),
                                                          config.getint('HyperParams', 'minibatchsize'),
                                                          config.getint('HyperParams', 'C'),
                                                          config.getint('HyperParams', 'D'),
                                                          2,
                                                          config.getint('HyperParams', 'annealingRate'),
                                                          config.getfloat('HyperParams', 'eta'),
                                                          config.getint('HyperParams', 'seed'),
                                                          config.getfloat('HyperParams', 'alpha'),
                                                          config.getint('HyperParams', 'X')))
        modelParameters.Init(trainTest)
        sgd = SGD()
        sgd.LearnParamsUsingSGD(trainTest, modelParameters)
        ScatterMatrix(modelParameters, ["good", "bad", "lame", "cool", "exciting"])

        #modelParameters = pickle_load('model.pkl')
        modelParameters = pickle_load('eta_2.0_model.pkl')

        print"model hyperparams:" + str(modelParameters.hyperParams)
        print "Predict input for - The movie was surprisingly __:"
        print "\t" + str(PredictInput(modelParameters, ["The", "movie", "was", "surprisingly"]))
        print "Predict input for - __ was really disappointing:"
        print "\t" + str(PredictInput(modelParameters, ["was", "really", "disappointing"]))
        print "Predict input for -Knowing that she __ was the best part:"
        print "\t" + str(PredictInput(modelParameters, ["Knowing", "that", "she", "was", "the" ,"best", "part"]))
        print "Solving analogy for - man is to woman as men is to:"
        print "\t" + str(AnalogySolver(modelParameters, "man", "woman", "men"))
        print "Solving analogy for - good is to great as bad is to:"
        print "\t" + str(AnalogySolver(modelParameters, "good", "great", "bad"))
        print "Solving analogy for -  warm is to cold as summer is to:"
        print "\t" + str(AnalogySolver(modelParameters, "warm", "cold", "summer"))


if __name__ == '__main__':
    main()
