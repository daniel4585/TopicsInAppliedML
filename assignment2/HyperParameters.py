class HyperParameters(object):
    def __init__(self, iterations, minibatchsize, C, D, K, annealingRate, eta, noiseDist, seed, alpha):
        self.minibatchsize = minibatchsize
        self.iterations = iterations
        self.C = C
        self.D = D
        self.K = K
        self.annealingRate = annealingRate
        self.eta = eta
        self.noiseDist = noiseDist
        self.seed = seed
        self.alpha = alpha


