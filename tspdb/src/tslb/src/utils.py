######################################################
#
# Utility functions
#
######################################################
import numpy as np
from numpy.linalg import eig
from math import log, e

# multinomial
def multinomial(n,p = [1/3,1/3,1/3]):
    final = []
    for i in range(n):
        result = np.random.multinomial(1, p)
        final.append(np.array(range(len(p)))[result == 1][0])
    return list(np.array(final).T)

def random_p(n=2):
    p = []
    for i in range(n):
        p0 = np.random.uniform(0,1,1)[0]
        p.append(p0)
    p = np.array(p)/np.sum(p)
    return list(p)

def get_p_tilda(uncompressed, n):
    p_tilda=[]
    for i in range(n):
        p_i = np.mean(np.array(uncompressed)==i)
        p_tilda.append(p_i)
    return p_tilda

def entropy(prob):
    ent = 0.
    for p in prob:
        if p < 1e-200:
            p = 1e-200
        ent = ent - p*log(p,2)
    return ent

# Markov
def random_P(n=2):
    P = np.zeros((n,n))
    for i in range(n):
        p=[]
        for j in range(n):
            p.append(np.random.uniform(0,1,1)[0])
        
        p = p/np.sum(p)
        P[i,:] = p
    return P

def get_next(this_obs, P):
    p = P[this_obs,:].flatten()
    next_obs = multinomial(1,p)
    return next_obs

def markov(len, P, initial = 0):
    this_obs = initial
    observations = [this_obs]
    for i in range(len):
        this_obs = get_next(this_obs, P)
        observations.append(this_obs[0])
    return observations

def entropy_rate(P):
    # P = transition matrix (n by n)
    # mu = asymptotic distribution (1 by n)
    n = P.shape[0]

    evals, evecs = eig(P.T)
    loc = np.where(np.abs(evals - 1.) < 0.0001)[0]
    stationary = evecs[:,loc].T
    mu = stationary / np.sum(stationary)
    mu = mu.real

    # print("evals")
    # print(evals)
    # print("evecs")
    # print(evecs)
    # print("stationary")
    # print(stationary)
    # print("mu")
    # print(mu)

    ent = 0
    for i in range(n):
        for j in range(n):
            ent = ent - mu[:,i] * P[i,j] * log(P[i,j],2)
    return ent[0]

# Fano's binary (alphabet size = 2)
def f(p):
    p = max(p,1e-200)
    return -p*log(p,2) - (1-p)*log(1-p,2)

def df(p):
    p = max(p,1e-200)
    return -log(p/(1-p),2) 

def f_inverse(H, a=0.001):
    # from entropy value, get p s.t. 0 < p < 0.5
    # a = accuracy
    p_hat = 0.25
    err = np.abs(f(p_hat) - H)
    while(err > a):
        err = np.abs(f(p_hat) - H)
        p_hat = p_hat - 0.01* (f(p_hat) - H) * df(p_hat)
        if (p_hat<0):
            p_hat = e-15
        if (p_hat>0.5):
            p_hat = 0.5
    return p_hat

# Fano's ternary (alphabet size = 3)
def g(p):
    return entropy([p,1-p]) + p

def dg(p):
    return -log(p/(1-p),2) + 1

def g_inverse(H, a=0.001):
    # from entropy value, get p s.t. 0 < p < 0.5
    # a = accuracy
    p_hat = 0.33
    err = np.abs(g(p_hat) - H)
    while(err > a):
        err = np.abs(g(p_hat) - H)
        p_hat = p_hat - 0.01* (g(p_hat) - H) * dg(p_hat)
        if (p_hat < 0):
            p_hat = 0
        if (p_hat > 2/3):
            p_hat = 2/3
    return p_hat

# Fano's (alphabet size = n)
def h(p, n):
    p = max(p,1e-200)
    return entropy([p,1-p]) + p * np.log2(n-1)

def dh(p, n):
    p = max(p,1e-200)
    return -log(p/(1-p),2) + np.log2(n-1)

def h_inverse(H, n, a=0.001):
    # from entropy value, get p s.t. 0 < p < 0.5
    # H = estimated entropy
    # n = alphabet size
    # a = accuracy

    p_hat = 0.33
    err = np.abs(h(p_hat, n) - H)
    count = 0
    while(err > a):
        if count > 1000:
            break
        err = np.abs(h(p_hat,n) - H)
        p_hat = p_hat - 0.01* (h(p_hat,n) - H) * dh(p_hat,n)
        if (p_hat < 0):
            p_hat = 0
        count += 1
    return p_hat

def get_error(seq1, seq2):
    return np.mean(seq1 != seq2)



# # Fano's, approximation
# def get_error_lower_bound_fano(H, n):
#     # H = estimated conditional entropy
#     # n = alphabet size (|X|)
#     return (H-1)/np.log2(n-1)


# def list_to_string(a):
#     return re.sub('\W+','', str(a) )

# def lzw_test(delta):
#     delta_string = list_to_string(delta)
#     delta_compressed = compress(delta_string)
#     delta_decompressed = decompress(delta_compressed)
#     ratio = len(delta_compressed)/len(delta_string)
#     error = np.sum(np.array([int(i) for i in delta_string]) != np.array([int(i) for i in delta_decompressed]))
    
#     print("compression using LZW")
#     print("original size   : ", len(delta))
#     print("compressed size : ", len(delta_compressed))
#     print("ratio           : ", ratio)
#     print("error           : ", error/len(delta_string))


