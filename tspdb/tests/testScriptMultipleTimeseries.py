#############################################################
#
# Single-Dimensional Time Series Imputation and Forecasting
#
# You need to ensure that this script is called from
# the tslib/ parent directory or tslib/tests/ directory:
#
# 1. python tests/testScriptmultipleTimeseries.py
# 2. python testScriptmultipleTimeseries.py
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

import tslib.src.data.generateHarmonics as gH
import tslib.src.data.generateTrend as gT
import tslib.src.data.generateARMA as gA
from  tslib.src.models.tsSVDModel import SVDModel
from tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils


def armaDataTest(timeSteps):

    arLags = [0.4, 0.3, 0.2]
    maLags = [0.5, 0.1]

    startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
    noiseMean = 0.0
    noiseSD = 1.0

    (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

    return (observedArray, meanArray)

def trendDataTest(timeSteps):

    dampening = 2.0*float(1.0/timeSteps)
    power = 0.35
    displacement = -2.5

    f1 = gT.linearTrendFn
    data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

    f2 = gT.logTrendFn
    data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    f3 = gT.negExpTrendFn
    t3 = gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    #plt.plot(t2)
    #plt.show()

    return data


def harmonicDataTest(timeSteps):

    sineCoeffs = [-2.0, 3.0]
    sinePeriods = [4.0, 10.0]

    cosineCoeffs = [-2.5]
    cosinePeriods = [12.0]

    data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
    #plt.plot(data)
    #plt.show()

    return data



# test for a multiple time series imputation and forecasting
def testMultipleTS():

    print("------------------- Test # 2 (Multiple TS). ------------------------")
    p = 1.0
    N = 50
    M = 400
    timeSteps = N*M
    
    # train/test split
    trainProp = 0.7
    M1 = int(trainProp * M)
    M2 = M - M1

    trainPoints = N*M1
    testPoints = N*M2

    key1 = 't1'
    key2 = 't2'
    key3 = 't3'
    otherkeys = [key2, key3]

    includePastDataOnly = True

    print("Generating data...")
    harmonicsTS = harmonicDataTest(timeSteps)
    trendTS = trendDataTest(timeSteps)
    (armaTS, armaMeanTS) = armaDataTest(timeSteps)

    meanTS = harmonicsTS + trendTS + armaMeanTS
    combinedTS = harmonicsTS + trendTS + armaTS

    combinedTS2 = (0.3 * combinedTS) + np.random.normal(0.0, 0.5, len(combinedTS))
    combinedTS3 = (-0.4 * combinedTS)

    #normalize the values to all lie within [-1, 1] -- helps with RMSE comparisons
    # can use the tsUtils.unnormalize() function to convert everything back to the original range at the end, if needed
    max1 = np.nanmax([combinedTS, combinedTS2, combinedTS3])
    min1 = np.nanmin([combinedTS, combinedTS2, combinedTS3])
    max2 = np.nanmax(meanTS)
    min2 = np.nanmin(meanTS)
    max = np.max([max1, max2])
    min = np.min([min1, min2])

    combinedTS = tsUtils.normalize(combinedTS, max, min)
    combinedTS2 = tsUtils.normalize(combinedTS2, max, min)
    combinedTS3 = tsUtils.normalize(combinedTS3, max, min)
    meanTS = tsUtils.normalize(meanTS, max, min)

    # produce timestamps
    timestamps = np.arange('2017-09-10 20:30:00', timeSteps, dtype='datetime64[1m]') # arbitrary start date

    # split the data
    trainDataMaster = combinedTS[0:trainPoints] # need this as the true realized values for comparisons later
    trainDataMaster2 = combinedTS2[0:trainPoints] 
    trainDataMaster3 = combinedTS3[0:trainPoints] 

    meanTrainData = meanTS[0:trainPoints] # this is only needed for various statistical comparisons later

    # randomly hide training data
    (trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster), p)
    (trainData2, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster2), p)
    (trainData3, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster3), p)

    # now further hide consecutive entries for a very small fraction of entries in the eventual training matrix
    (trainData, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData), 0.95, int(M1 * 0.25), M1)
    (trainData2, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData2), 0.95, int(M1 * 0.25), M1)
    (trainData3, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData3), 0.95, int(M1 * 0.25), M1)

    # once we have interpolated, pObservation should be set back to 1.0
    pObservation = 1.0

    # interpolating Nans with linear interpolation
    #trainData = tsUtils.nanInterpolateHelper(trainData)
    #trainData2 = tsUtils.nanInterpolateHelper(trainData2)
    #trainData3 = tsUtils.nanInterpolateHelper(trainData3)

    # test data and hidden truth
    testData = combinedTS[-1*testPoints: ]
    testData2 = combinedTS2[-1*testPoints: ]
    testData3 = combinedTS3[-1*testPoints: ]

    meanTestData = meanTS[-1*testPoints: ] # this is only needed for various statistical comparisons

    # time stamps
    trainTimestamps = timestamps[0:trainPoints]
    testTimestamps = timestamps[-1*testPoints: ]

    # create pandas df    
    trainMasterDF = pd.DataFrame(index=trainTimestamps, data={key1: trainDataMaster, key2: trainDataMaster2, key3: trainDataMaster3}) # needed for reference later
    trainDF = pd.DataFrame(index=trainTimestamps, data={key1: trainData, key2: trainData2, key3: trainData3})
    meanTrainDF = pd.DataFrame(index=trainTimestamps, data={key1: meanTrainData})

    testDF = pd.DataFrame(index=testTimestamps, data={key1: testData, key2: testData2, key3: testData3})
    meanTestDF = pd.DataFrame(index=testTimestamps, data={key1: meanTestData})

    # train the model
    print("Training the model (imputing)...")
    nbrSingValuesToKeep = 5
    mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=otherkeys, includePastDataOnly=includePastDataOnly)
    
    # uncomment below to run the ALS algorithm ; comment out the above line
    #mod = ALSModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, otherSeriesKeysArray=otherkeys, includePastDataOnly=True)
    mod.fit(trainDF)

    # imputed + denoised data 
    imputedDf = mod.denoisedDF()

    print(" RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
    print(" RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

    print("Forecasting (#points = %d)..." %len(testDF))

    # test data is used for point-predictions
    otherTSPoints = N
    if (includePastDataOnly == True):
        otherTSPoints = N - 1
    forecastArray = []
    for i in range(0, len(testDF)):
        
        pastPointsPrediction = np.zeros(N - 1) # for the time series of interest, we only use the past N - 1 points
        
        # first fill in the time series of interest
        j = 0
        if (i < N - 1):   # the first prediction uses the end of the training data
            while (j < N - 1 - i):
                pastPointsPrediction[j] = trainMasterDF[key1].values[len(trainDF) - (N - 1 - i) + j]
                j += 1

        if (j < N - 1): # use the new test data
            pastPointsPrediction[j:] = testDF[key1].values[i - (N - 1) + j:i] 

        # now fill in the other series
        otherSeriesDataDict = {}
        for key in otherkeys:
            pastPointsOthers = np.zeros(otherTSPoints) # need an appropriate length vector of past points for each series
            j = 0
            if (i < N - 1):   # the first prediction uses the end of the training data
                while (j < N - 1 - i):
                    pastPointsOthers[j] = trainMasterDF[key].values[len(trainDF) - (N - 1 - i) + j]
                    j += 1

            if (j < otherTSPoints): # use the new test data
                if (includePastDataOnly == True):
                    pastPointsOthers[j:] = testDF[key].values[i - (N - 1) + j:i] 
                else:
                    pastPointsOthers[j:] = testDF[key].values[i - (N - 1) + j:i + 1] 

            otherSeriesDataDict.update({key: pastPointsOthers})

        otherKeysToSeriesDFNew = pd.DataFrame(data=otherSeriesDataDict)
        keyToSeriesDFNew = pd.DataFrame(data={key1: pastPointsPrediction})

        prediction = mod.predict(otherKeysToSeriesDFNew, keyToSeriesDFNew, bypassChecks=False)
        forecastArray.append(prediction)

    print(" RMSE (prediction vs mean) = %f" %tsUtils.rmse(meanTestDF[key1].values, forecastArray))
    print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF[key1].values, forecastArray))

    print("Plotting...")
    plt.plot(np.concatenate((trainMasterDF[key1].values, testDF[key1].values), axis=0), color='gray', label='Observed')
    plt.plot(np.concatenate((meanTrainDF[key1].values, meanTestDF[key1].values), axis=0), color='red', label='True Means')
    plt.plot(np.concatenate((imputedDf[key1].values, forecastArray), axis=0), color='blue', label='Forecasts')
    plt.axvline(x=len(trainDF), linewidth=1, color='black', label='Training End')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.title('Single Time Series (ARMA + Periodic + Trend) - $p = %.2f$' %p)
    plt.show()

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    testMultipleTS()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":


    main()
