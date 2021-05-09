######################################################
#
# A wrapper for the SVD implementation of choice
#
######################################################
import numpy as np
from tspdb.src import tsUtils

class SVDWrapper:

    def __init__(self, matrix, method='numpy', threshold = 0.90):
        if (type(matrix) != np.ndarray):
            raise Exception('SVDWrapper required matrix to be of type np.ndarray')

        self.methods = ['numpy']

        self.matrix = matrix
        self.U = None
        self.V = None
        self.s = None
        self.next_sigma = 0
        self.threshold = threshold
        (self.N, self.M) = np.shape(matrix)

        if (method not in self.methods):
            print("The methods specified (%s) if not a valid option. Defaulting to numpy.linalg.svd" %method)
            self.method = 'numpy'

        else:
            self.method = method

    # perform the SVD decomposition
    # method will set the self.U and self.V singular vector matrices and the singular value array: self.s
    # U, s, V can then be access separately as attributed of the SVDWrapper class
    def decompose(self):
        # default is numpy's linear algebra library
        (self.U, self.s, self.V) = np.linalg.svd(self.matrix, full_matrices=False)
        
        # S = np.cumsum(self.s**2)
        # S = S/S[-1]
        # k = np.argmax(S>self.threshold)+1
        b = self.N/self.M
        omega = 0.56*b**3-0.95*b**2+1.43+1.82*b
        thre = omega*np.median(self.s)
        k = max(len(self.s[self.s>thre]), 1)
        # correct the dimensions of V
        self.V = self.V.T
        return k
    # get the top K singular values and corresponding singular vector matrices
    def decomposeTopK(self, k):

        # if k is 0 or less, just return empty arrays
        if k is not None:
            if (k < 1):
                return ([], [], [])

            # if k > the max possible singular values, set it to be that value
            elif (k > np.min([self.M, self.N])):
                k = np.min([self.M, self.N])

        if ((self.U is None) | (self.V is None) | (self.s is None)):
            est_k = self.decompose() # first perform the full decomposition
        if k is None:
            k = est_k

        if k < len(self.s)-1: self.next_sigma = self.s[k]
        else: self.next_sigma  = 0
        
        sk = self.s[0:k]
        Uk = self.U[:, 0:k]
        Vk = self.V[:, 0:k]

        return (sk, Uk, Vk)

    # get the matrix reconstruction using top K singular values
    # if returnMatrix = True, then return the actual matrix, else return sk, Uk, Vk
    def reconstructMatrix(self, kSingularValues, returnMatrix=False):

        (sk, Uk, Vk) = self.decomposeTopK(kSingularValues)
        if (returnMatrix == True):
            return tsUtils.matrixFromSVD(sk, Uk, Vk)
        else:
            return (sk, Uk, Vk)



