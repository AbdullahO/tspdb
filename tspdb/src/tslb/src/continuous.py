######################################################
#
# Utility functions for discretizing continuous distribution
#
######################################################
import numpy as np
import pandas as pd

# # uniform
# def entropy_uniform(b):
#     return math.log(b)
# def get_sequence_uniform(b, size=10000000):
#     return np.random.uniform(0, b, size=size)

# # uniform_sym
# def entropy_uniform_sym(b):
#     return math.log(2.*b)
# def get_sequence_uniform_sym(b, size=10000000):
#     return np.random.uniform(-b, b, size=size)

# # beta
# def B(a, b):
#     return math.gamma(a)*math.gamma(b)/math.gamma(a+b)
# def entropy_beta(alpha, beta):
#     return math.log(B(alpha,beta)) - (alpha-1)*(digamma(alpha)-digamma(alpha+beta)) - (beta-1)*(digamma(beta)-digamma(alpha+beta))
# def get_sequence_beta(alpha, beta, size=10000000):
#     return np.random.beta(alpha, beta, size=size)

# # triangular
# def entropy_tri(b):
#     return 0.5+math.log(b/2)
# def get_sequence_tri(b, size=10000000):
#     return np.random.triangular(0,b,1,size)

# # exponential
# def entropy_exp(lmbda):
#     return -math.log(lmbda)+1
# def get_sequence_exp(lmbda, size=10000000):
#     return np.random.exponential(1/lmbda, size=size)

# # normal
# def entropy_normal(sigma):
#     return math.log(sigma*np.sqrt(2*math.pi*math.e))
# def get_sequence_normal(sigma, size=10000000):
#     return np.random.normal(0, sigma**2 , size=size)


def get_sequence(dist, param, size=10000000):
    if dist=="uniform":
        # uniform([0,param])
        return np.random.uniform(0, param, size=size)
    if dist=="uniform_sym":
        # uniform ([-param, +param])
        return np.random.uniform(-param, param, size=size)
    if dist=="beta":
        # param = [alpha, beta]
        return np.random.beta(param[0], param[1], size=size)
    if dist=="tri":
        # param = mode
        return np.random.triangular(0,param,1,size)
    if dist=="exp":
        # param = lambda
        return np.random.exponential(1/param, size=size)
    if dist=="normal":
        # param = sigma
        return np.random.normal(0, param**2 , size=size)

def discretize(seq, bins):
    # seq: numpu 1d array, real-valued sequence
    # bins = 2**k = number of bins
    # discretized_seq, categories = pd.cut(np.hstack((seq, [0, 1])), bins, labels=np.arange(0, bins), retbins=True)
    discretized_seq, categories = pd.cut(seq, bins, labels=np.arange(0, bins), retbins=True)

    return np.array(discretized_seq), categories

def cut(seq, categories):
    return pd.cut(seq, categories, labels=np.arange(0, len(categories)-1), retbins=True)

# def discretized_p(dist, param, n):
#     len = 10000000
#     samples, categories = discretize(get_sequence(dist, param, len),n)
#     p=[]
#     for i in range(n):
#         p.append(np.mean(samples==i))
#     return p

def glm(init=0, a=1, length=100):
    # gaussian linear model
    # X(t+1) = a*X(t) + z, z~N(0,1)
    x = init

    seq=[]
    seq.append(init)
    for i in range(length-1):
        # print(i, ":", x)
        z = np.random.normal(0,1,size=1)[0]
        x = a*x + z
        seq.append(x)

    return seq

# def gaussian(x, mu=0, sigma2=1):
#     ''' f takes in a mean and squared variance, and an input x
#        and returns the gaussian value.'''
#     coefficient = 1.0 / sqrt(2.0 * pi *sigma2)
#     exponential = exp(-0.5 * (x-mu) ** 2 / sigma2)
#     return coefficient * exponential

# def update(mean1, var1, mean2, var2):
#     ''' This function takes in two means and two squared variance terms,
#         and returns updated gaussian parameters.'''
#     # Calculate the new parameters
#     new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
#     new_var = 1/(1/var2 + 1/var1)
#     return [new_mean, new_var]

# def predict(mean1, var1, mean2, var2):
#     ''' This function takes in two means and two squared variance terms,
#         and returns updated gaussian parameters, after motion.'''
#     # Calculate the new parameters
#     new_mean = mean1 + mean2
#     new_var = var1 + var2
#     return [new_mean, new_var]









