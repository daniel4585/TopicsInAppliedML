class Lambda(object):

    def __init__(self, lambda_u, lambda_v, lambda_b_u, lambda_b_v):
        super(Lambda, self).__init__()
        self.lambda_b_u = lambda_b_u
        self.lambda_b_v = lambda_b_v
        self.lambda_v = lambda_v
        self.lambda_u = lambda_u
