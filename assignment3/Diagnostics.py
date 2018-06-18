from Hyperparams import Hyperparams
import matplotlib.pyplot as plt

def plot_TrainTestError(hyperparams, outputFile="results.txt"):
    trainLoss = []
    testLoss = []
    with open("output/" + outputFile) as f:
        content = f.readlines()
        for i, x in enumerate(content):
            if i % 2 == 0:
                trainLoss.append(float(x.split(':')[1]))
            if i % 2 == 1:
                testLoss.append(float(x.split(':')[1]))

    line1, = plt.plot(range(0, hyperparams.numOfTrees), trainLoss, label="Train Loss")
    line2, = plt.plot(range(0, hyperparams.numOfTrees), testLoss, label="Test Loss")
    plt.suptitle("Loss as a function of number of iterations. \n Model hyper params:" + str(hyperparams))
    plt.legend(handles=[line1, line2])
    plt.xlabel("# of Trees")
    plt.ylabel("Mean Loss")
    plt.show()


def plot_varientParam(hyperParams, x_axis, finalTrainLosses, finalTestLosses, total_time, title, x_axis_label):
    f, (ax1, ax2) = plt.subplots(1, 2)
    line1, = ax1.plot(x_axis, finalTrainLosses, label="Train Log Likelihood")
    line2, = ax1.plot(x_axis, finalTestLosses, label="Test Log Likelihood")
    f.suptitle(" Model hyper params:" + str(hyperParams))
    ax1.set_title("Final Loss as a function of " + title)
    ax1.legend(handles=[line1, line2])
    ax1.set(xlabel=x_axis_label, ylabel='Mean Loss')


    line1, = ax2.plot(x_axis, total_time)
    ax2.set_title("Train Time as a function of " + title)
    ax1.set(xlabel=x_axis_label, ylabel='Time in S')
    plt.show()
