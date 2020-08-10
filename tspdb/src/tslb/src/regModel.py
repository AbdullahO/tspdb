######################################################
#
# Regression Model
#
######################################################
import numpy as np
from sklearn.linear_model import LinearRegression

from tspdb.src.tslb.src.lzw import *
from tspdb.src.tslb.src.utils import *


class regModel():
    # n = alphabet size
    # size = sequence length
    # samples = number of samples to collect
    def __init__(self, n, size, samples):
        self.n = n
        self.size = size
        self.samples = samples

        # ratio = ratio list to fit reg
        # entropy = entropy list to fit entropy
        self.ratio = None
        self.entropy = None
        self.reg = None
        self.reg_inv = None

    def fit(self):
        dictionary = {i : chr(i) for i in range(self.n)}
        
        ratio_list =[]
        true_entropy = []
        
        probabilities=[]
        for i in range(self.samples):
            probabilities.append(random_p(self.n))
                
        for p in probabilities:
            true_entropy.append(entropy(p))
            
            uncompressed =str()        
            uncomp_numbers = multinomial(self.size, p)
            for i in uncomp_numbers:
                uncompressed = uncompressed + dictionary[i]

            compressed = compress(uncompressed)
            compression_ratio = len(compressed)/len(uncompressed)
            ratio_list.append(compression_ratio)

        self.ratio = ratio_list
        self.entropy = true_entropy

        # linear regression
        self.reg = LinearRegression(fit_intercept=True).fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(self.ratio[:]))
        self.reg_inv = LinearRegression(fit_intercept=True).fit(np.array(self.ratio[:]).reshape(-1, 1), np.array(true_entropy[:]))
        score = self.reg.score(np.array(true_entropy[:]).reshape(-1, 1), np.array(self.ratio[:])).round(3)




    def get_entropy(self, compression_ratio):
        # mapping compression ratio to entropy
        ent = self.reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]


        return ent
