######################################################
#
# Utility functions
#
######################################################
import numpy as np
from sklearn.metrics import mean_squared_error
import copy
from numpy.linalg import qr  as qr

def unnormalize(arr, mean, std):
    return arr *std + mean
    
def updateSVD(D, uk, sk, vk):
    vk = vk.T
    m = vk.shape[1]
    d = m+D.shape[1]
    D_k = np.dot(np.dot(D.T, uk), np.diag(1 / sk))
    vkh = np.zeros([len(sk), d])
    vkh[:, :m] = vk
    vkh[:, m:d] = D_k.T

    return uk, sk, vkh.T


def updateSVD2(D, uk, sk, vk):
    vk = vk.T
    k,m = vk.shape
    n,p = D.shape
    # memory intensive? nxn dot nxp
    D_h = np.dot(np.eye(n)- np.dot(uk,uk.T),D)
    # Qr of n X p matrix ~ relatively easy
    Qd,Rd = qr(D_h)

    A_h = np.zeros([p+k,p+k])
    A_h[:k,:k] = np.diag(sk)
    A_h[:k,k:k+p] = np.dot(uk.T,D)
    A_h[k:k+p, k:k+p] = Rd
    # SVD of p+k X p+k matrix ~ relatively easy
    ui, si, vi = np.linalg.svd(A_h, full_matrices=False)
    uk_h = ui[:,:k]
    sk_h = si[:k]
    vk_h = vi[:k,:]

    sk_u = sk_h

    # matirx mult. n X (k+p) by (k+p) X k
    #uk_u = np.dot(np.concatenate((uk,Qd),1),uk_h)
    uk_u = np.zeros([n, k+p])
    uk_u[:,:k] = uk
    uk_u[:, k:k+p] = Qd
    uk_u = np.dot(uk_u,uk_h)

    vk_u = np.zeros([m+p,k+p])
    vk_u[:m,:k] = vk.T
    vk_u[m:m+p, k:k+p] = np.eye(p)

    vk_2 = np.dot(vk_u,vk_h.T)
    return uk_u, sk_u, vk_2

def arrayToMatrix(npArray, nRows, nCols):

    if (type(npArray) != np.ndarray):
        raise Exception('npArray is required to be of type np.ndarray')

    if (nRows * nCols != len(npArray)):
        raise Exception('(nRows * nCols) must equal the length of npArray')

    return np.reshape(npArray, (nCols, nRows)).T


def matrixFromSVD(sk, Uk, Vk, soft_threshold = 0, probability=1.0):
    return (1.0/probability) * np.dot(Uk, np.dot(np.diag(sk), Vk.T))


def pInverseMatrixFromSVD(sk, Uk, Vk, soft_threshold=0,probability=1.0):
    s = copy.deepcopy(sk)
    s = s - soft_threshold

    for i in range(0, len(s)):
        if (s[i] > 0.0):
            s[i] = 1.0/s[i]

    p = 1.0/probability
    return matrixFromSVD(s, Vk, Uk, probability=p)


def rmse(array1, array2):
    return np.sqrt(mean_squared_error(array1, array2))


def rmseMissingData(array1, array2):

    if (len(array1) != len(array2)):
        raise Exception('lengths of array1 and array2 must be the same.')

    subset1 = []
    subset2 = []
    for i in range(0, len(array1)):
        if np.isnan(array1[i]):
            subset1.append(array1[i])
            subset2.append(array2[i])

    return rmse(subset1, subset2)


# def normalize(array, max, min, pos = False):
#     """

#     :param array:
#     :param max:
#     :param min:
#     :param pos: if true, normalize between 0 and 1
#     :return:
#     """


#     if pos:
#         array = (array - min)/(max-min)
#     else:
#         diff = 0.5 * (min + max)
#         div = 0.5 * (max - min)
#         array = (array - diff) / div
#     return array

# def unnormalize(array, max, min, pos = False):


#     if pos:
#         array = array *(max-min) + min
#     else:
#         diff = 0.5 * (min + max)
#         div = 0.5 * (max - min)

#         array = (array * div) + diff
#     return array


def randomlyHideValues(array, pObservation):

    count = 0
    for i in range(0, len(array)):
        if (np.random.uniform(0, 1) > pObservation):
            array[i] = np.nan
            count +=1 

    p_obs = float(count)/float(len(array))
    return (array, 1.0 - p_obs)

# chooses rows of the matrix according to pObservationRow
# hide stretches of data with the longestStretch being the max entries hidden in a row
# gap should ideally be the number of columns of the matrix this array will be converted in to
def randomlyHideConsecutiveEntries(array, pObservationRow, longestStretch, gap):

    n = len(array)
    valuesToHide = int((1.0 - pObservationRow) * n)

    count = 0
    countStart = 0
    i = 0
    while (i < n):
        # decide if this point is the start of a randomly missing run
        if (np.random.uniform(0, 1) > pObservationRow):
            countStart +=1

            # now decide how many consecutive values go missing and where to start
            toHide = longestStretch #int(np.random.uniform(0, 1) * longestStretch)
            startingIndex = i + int(np.random.uniform(0, 1) * (gap - toHide))

            if (toHide + startingIndex >  (i + gap)):
                toHide = (i + gap) - startingIndex

            array[startingIndex: startingIndex + toHide] = np.nan * np.zeros(toHide)
            
            count += toHide

            valuesToHide -= toHide

            if (valuesToHide <= 0):
                break

        # ensure there is some space between consecutive runs
        i += gap

    p_obs = float(count)/float(n)

    return (array, 1.0 - p_obs)




#######################################################
# Testing 

# arr = [1,2.0,3.0,4,5,5,6,7,8,19, 29, 49]
# arr = np.array(arr)
# arr[0] = np.nan
# arr[8] = np.nan
# print(arr)
# arr = nanInterpolateHelper(arr)
# print(arr)


# N = 4
# T = 4
# data = np.random.normal(0.0, 10.0, N*T)
# #print(data)
# M = arrayToMatrix(data, N, T)

# # import algorithms.svdWrapper
# # from algorithms.svdWrapper import SVDWrapper as SVD
# # svdMod = SVD(M, method='numpy')
# # (sk, Uk, Vk) = svdMod.reconstructMatrix(4, returnMatrix=False)

# #M1 = matrixFromSVD(sk, Uk, Vk, probability=1.0)
# #print(np.mean(M - M1))

# (Uk, sk, Vk) = np.linalg.svd(M, full_matrices=False)
# Vk = Vk.T

# MA = np.linalg.pinv(M)
# MB = pInverseMatrixFromSVD(sk, Uk, Vk, probability=1.0)
# print(np.mean(MA - MB))

# M22 = matrixFromSVD(sk, Uk[0:-1, :], Vk, probability=1.0)

# M2 = pInverseMatrixFromSVD(sk, Uk[0:-1, :], Vk, probability=1.0)
# M4 = np.linalg.pinv(M)

# print(M2)
# print(M4)


# M3 = np.dot(np.dot(M, M2), M)
# print(np.mean(M - M3))