######################################################
#
# Alternating Least Squares
#
######################################################
import numpy as np
from tspdb.src import tsUtils

class ALSWrapper:

    def __init__(self, matrix, method='als'):
        if (type(matrix) != np.ndarray):
            raise Exception('ALSWrapper required matrix to be of type np.ndarray')

        self.methods = ['als']

        self.matrix = matrix

        (self.N, self.M) = np.shape(matrix)

        self.W = np.zeros([self.N, self.M])
        mask = np.isnan(self.matrix)
        self.W[mask == True] = 0.0
        self.W[mask == False] = 1.0
        self.W = self.W.astype(np.float64, copy=False)

        self.matrix[mask == True] = 0.0

        if (method not in self.methods):
            print("The methods specified (%s) if not a valid option. Defaulting to ALS" %method)
            self.method = 'als'

        else:
            self.method = method

    # run the ALS algorithm
    # k is the number of factors
    def decompose(self, k, lambda_, iterations, tol):

        middleVal = 0.5 * (np.max(self.matrix) + np.min(self.matrix))

        # initialize randomly
        U = middleVal * np.random.rand(self.N, k) 
        V = middleVal * np.random.rand(k, self.M)

        # fix max iterations
        maxIter = iterations

        pastError = np.inf
        for ii in range(maxIter):
            # first U matrix with V fixed
            for u, Wu in enumerate(self.W):
                left = np.linalg.pinv(np.dot(V, np.dot(np.diag(Wu), V.T)) + lambda_ * np.eye(k))
                right = np.dot(V, np.dot(np.diag(Wu), self.matrix[u].T))
                U[u] = np.dot(left, right).T

                    #np.linalg.solve(np.dot(V, np.dot(np.diag(Wu), V.T)) + lambda_ * np.eye(k),
                              # np.dot(V, np.dot(np.diag(Wu), self.matrix[u].T))).T

            # now V matrix with U fixed
            for i, Wi in enumerate(self.W.T):
                left = np.linalg.pinv(np.dot(U.T, np.dot(np.diag(Wi), U)) + lambda_ * np.eye(k))
                right = np.dot(U.T, np.dot(np.diag(Wi), self.matrix[:, i]))
                V[:,i] = np.dot(left, right).T

                #np.linalg.solve(np.dot(U.T, np.dot(np.diag(Wi), U)) + lambda_ * np.eye(k),
                          #       np.dot(U.T, np.dot(np.diag(Wi), self.matrix[:, i])))
            
            # compute MSE
            err = self.getError(self.matrix, U, V, self.W)

            # break if difference is less than tol
            deltaErr = np.abs(err - pastError)
            if (deltaErr < tol):
                break
            else:
                pastError = err

            if (ii%10 == 0):
                print("Iteration %d, Err = %0.4f, DeltaErr = %0.4f" %(ii+1, pastError, deltaErr))

        print('Total Iterations = %d' %(ii+1))
        return (U,V)
        


    # get the matrix reconstruction using k factors and missing data
    def reconstructMatrix(self, k, lambda_, returnMatrix=True, iterations=1000, tol=1e-6):

        (Uk, Vk) = self.decompose(k, lambda_, iterations, tol)
        if (returnMatrix == True):
            return np.dot(Uk, Vk)
        else:
            return (Uk, Vk)


    # MSE function for the ALS algorithm
    def getError(self, Q, U, V, W):
        return np.mean((W * (Q - np.dot(U, V)))**2)

# ##################################################
# # Test code

# import time
# from tslib.src.algorithms.svdWrapper import SVDWrapper
# import copy

# # generate a rank-k matrix
# N = 1000
# M = 500
# k = 2

# paramsN = np.random.rand(N)
# paramsM = np.random.rand(M)

# Y = np.ones([N, M])
# Y1 = np.ones([N, M])
# Y2 = np.ones([N, M])
# count = 0
# for i in range(0, N):
#     for j in range(0, M):
#         Y[i, j] = 10.0* (paramsN[i] + paramsM[j]) #+ np.random.normal(0.0, 0.5)

# # normalize to lie between [-1, 1]
# print(count)
# min = np.min(Y)
# max = np.max(Y)

# Y = tsUtils.normalize(Y, max, min)

# for i in range(0, N):
#     for j in range(0, M):
#         Y1[i, j] = Y[i, j]
#         Y2[i, j] = Y[i, j]

#         if (np.random.uniform() > 0.6):
#             Y1[i, j] = np.nan
#             Y2[i, j] = 0.0
#     if (np.random.uniform() > 0.7):
#         Y1[i, M/2:] = np.nan * np.zeros(M - M/2)
#         Y2[i, M/2:] = np.zeros(M - M/2)
#         count += 1

# middleVal = 0.5 * (np.max(Y) + np.min(Y))
# Y2[Y2 == 0.0] = middleVal
# print(np.linalg.matrix_rank(Y))
# # ALS

# t1 = time.time()

# ALS = ALSWrapper(copy.deepcopy(Y1))
# Yhat = ALS.reconstructMatrix(k, 0.0, returnMatrix=True, tol=1e-4)

# t2 = time.time()
# print("ALS:")
# print(tsUtils.rmse(Yhat, Y), t2-t1)
# print(np.linalg.matrix_rank(Yhat))

# # SVD
# t3 = time.time()
# svd1 = SVDWrapper(Y2, method='numpy')
# YhatSVD = svd1.reconstructMatrix(k, returnMatrix=True)
# t4 = time.time()
# print("SVD:")
# print(tsUtils.rmse(YhatSVD, Y), t4-t3)



