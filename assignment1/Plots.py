import math
import matplotlib.pyplot as plt


def plot_error(regularization, sgd_params):
    with open("output/SGD_train_error.txt") as f:
        content = f.readlines()
    sgd_train_error = [float(x.strip().split(':')[1]) for x in content]

    with open("output/SGD_test_error.txt") as f:
        content = f.readlines()
    sgd_test_error = [float(x.strip().split(':')[1]) for x in content]

    with open("output/ALS_train_error.txt") as f:
        content = f.readlines()
    als_train_error = [float(x.strip().split(':')[1]) for x in content]

    with open("output/ALS_test_error.txt") as f:
        content = f.readlines()
    als_test_error = [float(x.strip().split(':')[1]) for x in content]

    plt.subplot(121)
    plt.plot(range(len(sgd_train_error)), sgd_train_error, label="sgd_train_error")
    plt.plot(range(len(sgd_test_error)), sgd_test_error, label="sgd_test_error")
    plt.suptitle("SGD: " + str(regularization) + str(sgd_params))

    plt.subplot(122)
    plt.plot(range(len(als_train_error)), als_train_error, label="als_train_error")
    plt.plot(range(len(als_test_error)), als_test_error, label="als_test_error")
    plt.legend(loc='lower right')
    plt.suptitle("ALS: " + str(regularization))
    plt.show()


def plot_lambda(lambdas, rmses, mprs, sgd_params):
    lambdas = [math.log(x, 10) for x in lambdas]
    plt.subplot(121)
    plt.plot(lambdas, rmses)
    plt.xlabel('log(lambda)')
    plt.ylabel("RMSE")
    title = "Lambda train - SGD Params:" + str(sgd_params)
    plt.suptitle(title, fontsize=16)

    plt.subplot(122)
    plt.plot(lambdas, mprs)
    plt.xlabel('log(lambda)')
    plt.ylabel("MPR")
    plt.show()


def plot_dim(Ks, rmses, mprs, sgd_params):
    plt.subplot(121)
    plt.plot(Ks, rmses)
    plt.xlabel('# of Latent Dim ')
    plt.ylabel("RMSE")
    title = "#of Latent dim train - SGD Params:" + str(sgd_params)
    plt.suptitle(title, fontsize=16)

    plt.subplot(122)
    plt.plot(Ks, mprs)
    plt.xlabel('# of Latent Dim')
    plt.ylabel("MPR")
    plt.show()

def plot_dim_times(Ks, times, sgd_params):
    plt.subplot()
    plt.plot(Ks, times)
    plt.xlabel('# of Latent Dim ')
    plt.ylabel("Time in sec")
    title = "# of Latent dim train - SGD Params:" + str(sgd_params)
    plt.suptitle(title, fontsize=16)
    plt.show()
