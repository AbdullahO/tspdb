#############################################################
#
# Multi-Dimensional Robust Synthetic Control Tests 
# (based on SVD)
#
# Generates two metrics and compared the RMSE for forecasts
# for each metric using RSC against mRSC.
#
# Test are based on random data so it is advised to run
# several times. Also note that in this setting RSC is 
# also expected to do well. mRSC is expected to help but
# cannot be guaranteed to always be better.
#
# You need to ensure that this script is called from
# the tslib/ parent directory or tslib/tests/ directory:
#
# 1. python tests/testScriptMultiSynthControlSVD.py
# 2. python testScriptMultiSynthControlSVD.py
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src.synthcontrol.multisyntheticControl import MultiRobustSyntheticControl

def simpleFunctionOne(theta, rho):
    alpha = 0.7
    exp_term = np.exp((-1.0 *theta) - rho -  (alpha * theta * rho))

    exp_term2 = np.exp(-1.0 *alpha * theta * rho)

    p = 10.0 * float(1.0 / (1.0 + exp_term)) + 10.0/exp_term2

    return p

def simpleFunctionTwo(theta, rho):
    alpha = 0.5
    exp_term = np.exp((-1.0 *theta) - rho -  (alpha * theta * rho))

    p = 10.0 * float(1.0 / (1.0 + exp_term))

    return p

def generateDataMatrix(N, T, rowRank, colRank, genFunction, rowParams, colParams):

    matrix = np.zeros([N, T])
    for i in range(0, N):
        for j in range(0, T):
            matrix[i, j] = genFunction(rowParams[i], colParams[j])

    return matrix

def generateFirstRow(matrix, weights):

    (N, T) = np.shape(matrix)
    assert(len(weights) == N)
    weights = weights.reshape([N, 1])

    weights = weights/np.sum(weights)

    return np.dot(weights.T, matrix)


def generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunction, trueWeights, rowParams, colParams):

    matrix = generateDataMatrix(N, T, rowRank, colRank, genFunction, rowParams, colParams)

    firstRow = generateFirstRow(matrix, trueWeights)

    meanMatrix = np.zeros([N+1, T]) #np.concatenate([firstRow, matrix], axis=0)
    meanMatrix[0, :] = firstRow
    meanMatrix[1:, :] = matrix
    
    #print(np.linalg.matrix_rank(meanMatrix))

    noiseMatrix = np.random.normal(0.0, 1.0, [N+1, T])
    #print(np.linalg.matrix_rank(noiseMatrix))

    observationMatrix = meanMatrix + noiseMatrix
    #print(np.linalg.matrix_rank(observationMatrix))

    # convert to dataframes
    trainingDict = {}
    testDict = {}

    meanTrainingDict = {}
    meanTestDict = {}
    for i in range(0, N+1):

        trainingDict.update({str(i): observationMatrix[i, 0:TrainingEnd]})
        meanTrainingDict.update({str(i): meanMatrix[i, 0:TrainingEnd]})

        testDict.update({str(i): observationMatrix[i, TrainingEnd:]})
        meanTestDict.update({str(i): meanMatrix[i, TrainingEnd:]})

    trainDF = pd.DataFrame(data=trainingDict)
    testDF = pd.DataFrame(data=testDict)

    meanTrainDF = pd.DataFrame(data=meanTrainingDict)
    meanTestDF = pd.DataFrame(data=meanTestDict)

    #print(np.shape(trainDF), np.shape(testDF))
    #print(np.shape(meanTrainDF), np.shape(meanTestDF))

    return (observationMatrix, meanMatrix, trainDF, testDF, meanTrainingDict, meanTestDict)


def rankComparison():
    N = 100
    T = 120
    TrainingEnd = 100
    rowRank = 200
    colRank = 200

    # generate metric matrices
    genFunctionOne = simpleFunctionOne
    genFunctionTwo = simpleFunctionTwo

    trueWeights = np.random.uniform(0.0, 1.0, N)
    trueWeights = trueWeights/np.sum(trueWeights)

    thetaArrayParams = np.random.uniform(0.0, 1.0, rowRank)
    rhoArrayParams = np.random.uniform(0.0, 1.0, colRank)

    rowParams = np.random.choice(thetaArrayParams, N)
    colParams = np.random.choice(rhoArrayParams, T)

    # metric 1
    (observationMatrix, meanMatrix, trainDF, testDF, meanTrainingDict, meanTestDict) = generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunctionOne, trueWeights, rowParams, colParams)
    
    # metric 2
    (observationMatrix2, meanMatrix2, trainDF2, testDF2, meanTrainingDict2, meanTestDict2) = generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunctionTwo, trueWeights, rowParams, colParams)
    

    thetaArrayParams = np.random.uniform(0.0, 1.0, rowRank)
    rhoArrayParams = np.random.uniform(0.0, 1.0, colRank)

    rowParams = np.random.choice(thetaArrayParams, N)
    colParams = np.random.choice(rhoArrayParams, T)

    # metric 3
    (observationMatrix3, meanMatrix3, trainDF3, testDF3, meanTrainingDict3, meanTestDict3) = generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunctionTwo, trueWeights, rowParams, colParams)
    
    # concatenation
    matrixA = np.zeros([N+1, 2*T])
    matrixA[:, 0:T] = meanMatrix
    matrixA[:, T: ] = meanMatrix2

    u, s, v = np.linalg.svd(meanMatrix, full_matrices=False)
    u, s_, v = np.linalg.svd(meanMatrix2, full_matrices=False)

    u, sA, v = np.linalg.svd(matrixA, full_matrices=False)

    # print(np.linalg.matrix_rank(meanMatrix))
    # print(np.linalg.matrix_rank(meanMatrix2))
    # print(np.linalg.matrix_rank(meanMatrix3))
    # print(np.linalg.matrix_rank(matrixA))


    k = 20
    plt.plot(range(0, k), s[0:k], color='magenta', label='metric1')
    plt.plot(range(0, k), s_[0:k], color='black', label='metric2')
    plt.plot(range(0, k), sA[0:k], '-x', color='red', label='combined')
    plt.xlabel('Singular Value Index (largest to smallest)')
    plt.ylabel('Singular Value')
    plt.title('Diagnostic: Rank Preservation Property')
    
    legend = plt.legend(loc='lower right', shadow=True)
    plt.show()




def runAnalysis(N, T, TrainingEnd, rowRank, colRank):

    # generate metric matrices
    genFunctionOne = simpleFunctionOne
    genFunctionTwo = simpleFunctionTwo

    trueWeights = np.random.uniform(0.0, 1.0, N)
    trueWeights = trueWeights/np.sum(trueWeights)

    thetaArrayParams = np.random.uniform(0.0, 1.0, rowRank)
    rhoArrayParams = np.random.uniform(0.0, 1.0, colRank)

    rowParams = np.random.choice(thetaArrayParams, N)
    colParams = np.random.choice(rhoArrayParams, T)

    # metric 1
    (observationMatrix1, meanMatrix1, trainDF1, testDF1, meanTrainingDict1, meanTestDict1) = generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunctionOne, trueWeights, rowParams, colParams)
    
    # metric 2
    (observationMatrix2, meanMatrix2, trainDF2, testDF2, meanTrainingDict2, meanTestDict2) = generateOneMetricMatrix(N, T, TrainingEnd, rowRank, colRank, genFunctionTwo, trueWeights, rowParams, colParams)
    

    keySeriesLabel = '0'
    otherSeriesLabels = []
    for ind in range(1, N+1):
        otherSeriesLabels.append(str(ind))

    # RSC analysis
    singvals = 8

    ############################
    #### RSC for metric 1
    rscmodel1 = RobustSyntheticControl(keySeriesLabel, singvals, len(trainDF1), probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=otherSeriesLabels)

    # fit the model
    rscmodel1.fit(trainDF1)
    predictionsRSC1 = rscmodel1.predict(testDF1)

    
    rscRMSE1 = np.sqrt(np.mean((predictionsRSC1 - meanTestDict1[keySeriesLabel])**2))
    #print("\n\n *** RSC rmse1:")
    #print(rscRMSE1)

    ############################
    ##### RSC for metric 2
    rscmodel2 = RobustSyntheticControl(keySeriesLabel, singvals, len(trainDF2), probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=otherSeriesLabels)

    # fit the model
    rscmodel2.fit(trainDF2)
    predictionsRSC2 = rscmodel2.predict(testDF2)

    
    rscRMSE2 = np.sqrt(np.mean((predictionsRSC2 - meanTestDict2[keySeriesLabel])**2))
    #print("\n\n *** RSC rmse2:")
    #print(rscRMSE2)

    ############################
    ####  multi RSC model (combined) --
    relative_weights = [1.0, 1.0]

    # instantiate the model
    mrscmodel = MultiRobustSyntheticControl(2, relative_weights, keySeriesLabel, singvals, len(trainDF1), probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=otherSeriesLabels)
    
    # fit
    mrscmodel.fit([trainDF1, trainDF2])
    
    # predict
    combinedPredictionsArray = mrscmodel.predict([testDF1[otherSeriesLabels], testDF2[otherSeriesLabels]])

    # split the predictions for the metrics
    predictionsmRSC_1 = combinedPredictionsArray[0]
    predictionsmRSC_2 = combinedPredictionsArray[1]

    # compute RMSE
    mrscRMSE1 = np.sqrt(np.mean((predictionsmRSC_1 - meanTestDict1[keySeriesLabel])**2))
    mrscRMSE2 = np.sqrt(np.mean((predictionsmRSC_2 - meanTestDict2[keySeriesLabel])**2))

    #print("\n\n *** mRSC rmse1:")
    #print(mrscRMSE1)

    #print("\n\n *** mRSC rmse2:")
    #print(mrscRMSE1)

    return ({"rsc1": rscRMSE1,
            "rsc2":  rscRMSE2,
            "mrsc1": mrscRMSE1,
            "mrsc2": mrscRMSE2})

def main():

    # diagnostic test for rank preservation (see paper referenced)
    rankComparison()

    rowRank = 10
    colRank = 10

    rsc1 = []
    rsc1A = []
    rsc2 = []
    rsc2A = []

    mrsc1 = []
    mrsc1A = []
    mrsc2 = []
    mrsc2A = []

    # simulating many random tests and varying matrix sizes
    N_array = [50, 100, 150, 200, 250, 300]
    for N in N_array:
        print("**********************************************************")
        print(N)
        print("**********************************************************")
        for T in [30]: 
            TrainingEnd = int(0.75*T)

            rsc1_array = []
            rsc1A_array = []
            rsc2_array = []
            rsc2A_array = []

            mrsc1_array = []
            mrsc1A_array = []
            mrsc2_array = []
            mrsc2A_array = []

            for iter in range(0, 20):
                resDict = runAnalysis(N, T, TrainingEnd, rowRank, colRank)

                rsc1_array.append(resDict['rsc1'])
                rsc2_array.append(resDict['rsc2'])
                mrsc1_array.append(resDict['mrsc1'])
                mrsc2_array.append(resDict['mrsc2'])

            rsc1.append(np.mean(rsc1_array))
            rsc2.append(np.mean(rsc2_array))

            mrsc1.append(np.mean(mrsc1_array))
            mrsc2.append(np.mean(mrsc2_array))

    print("====================================")
    print("====================================")
    print("Metric # 1:")
    print("mRSC - RSC:")
    for i in range(0, len(N_array)):
        print(i, mrsc1[i] - rsc1[i])

    print("Metric # 2:")
    print("mRSC - RSC:")
    for i in range(0, len(N_array)):
        print(i, mrsc2[i] - rsc2[i])
    print("====================================")
    print("====================================")


    print("====================================")
    print("====================================")
    print("Metric # 1:")
    print("mRSC, RSC:")
    for i in range(0, len(N_array)):
        print(i, mrsc1[i], rsc1[i])

    print("Metric # 2:")
    print("mRSC, RSC,:")
    for i in range(0, len(N_array)):
        print(i, mrsc2[i], rsc2[i])
    print("====================================")
    print("====================================")

        
    # plotting
    plt.plot(N_array, mrsc1, color='r', label='mRSC (metricA)')
    plt.plot(N_array, mrsc2, color='orange', label='mRSC (metricB)')
    plt.plot(N_array, rsc1, '-.', color='blue', label='RSC (metricA)')
    plt.plot(N_array, rsc2, '--x',color='magenta', label='RSC (metricB)')
    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.title('mRSC vs RSC for metricA and metricB')
    
    legend = plt.legend(loc='upper right', shadow=True)
    plt.show()

if __name__ == "__main__":
    main()




