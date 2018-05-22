import math
import matplotlib.pyplot as plt

def plot_logLiklihood(hyperparms):

    iteration = []
    trainLL = []
    testLL = []
    with open("output/loglikelihood.txt") as f:
        content = f.readlines()
        for i, x in enumerate(content):
            if i % 3 == 0:
                iteration.append(int(x.split(':')[1]))
            if i % 3 == 1:
                trainLL.append(float(x.split(':')[1]))
            if i % 3 == 2:
                testLL.append(float(x.split(':')[1]))
    line1, = plt.plot(iteration, trainLL, label="Train Log Likelihood")
    line2, = plt.plot(iteration, testLL, label="Test Log Likelihood")
    plt.suptitle("Logliklihood as a function of number of iterations. \n Model hyper params:" + str(hyperparms))
    plt.legend(handles=[line1, line2])
    plt.xlabel("# of iterations")
    plt.ylabel("Mean LogLikelihood")
    plt.show()

def plot_varientParam(hyperParams, x_axis, finalTrainLogLiklihood, finalTestLogLiklihood, total_time, title, x_axis_label):
    f, (ax1, ax2) = plt.subplots(1, 2)
    line1, = ax1.plot(x_axis, finalTrainLogLiklihood, label="Train Log Likelihood")
    line2, = ax1.plot(x_axis, finalTestLogLiklihood, label="Test Log Likelihood")
    f.suptitle(" Model hyper params:" + str(hyperParams))
    ax1.set_title("Logliklihood as a function of " + title)
    ax1.legend(handles=[line1, line2])
    ax1.set(xlabel=x_axis_label, ylabel='Mean LogLikelihood')


    line1, = ax2.plot(x_axis, total_time, label="Train Log Likelihood")
    ax2.set_title("Train Time as a function of " + title)
    ax1.set(xlabel=x_axis_label, ylabel='Time in S')
    plt.show()
