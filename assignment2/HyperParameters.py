class HyperParameters(object):
    def __init__(self, iterations, minibatchsize, C, D, K, annealingRate, eta, seed, alpha, X):
        self.minibatchsize = minibatchsize
        self.iterations = iterations
        self.C = C
        self.D = D
        self.K = K
        self.annealingRate = annealingRate
        self.eta = eta
        self.seed = seed
        self.alpha = alpha
        self.X = X

    def __str__(self):
        return "minibatchsize:" + str(self.minibatchsize) + ", iterations:" + str(self.iterations) + ", C:" + str(self.C) +\
               ", D:" + str(self.D) + ", K:" + str(self.K) + ", annealingRate:" + str(self.annealingRate) + ", eta:" + str(self.eta) +\
         ", seed:" + str(self.seed) + ", alpha:" + str(self.alpha) + ", X:" + str(self.X)
