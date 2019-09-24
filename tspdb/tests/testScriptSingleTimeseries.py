#############################################################
#
# Single-Dimensional Time Series Imputation and Forecasting
#
# You need to ensure that this script is called from
# the tslib/ parent directory or tslib/tests/ directory:
#
# 1. python tests/testScriptSingleTimeseries.py
# 2. python testScriptSingleTimeseries.py
#
#############################################################
import sys, os

sys.path.append("../..")
sys.path.append("C:\\Users\Abdul\OneDrive\Documents\GitHub\\tslib")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src.data import generateHarmonics as gH
from  tslib.src.data import generateTrend as gT
import tslib.src.data.generateARMA as gA
from  tslib.src.models.tsSVDModel import SVDModel
from  tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils


def armaDataTest(timeSteps):
    arLags = [0.4, 0.3, 0.2]
    maLags = [0.5, 0.1]

    startingArray = np.zeros(np.max([len(arLags), len(maLags)]))  # start with all 0's
    noiseMean = 0.0
    noiseSD = 1.0

    (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

    return (observedArray, meanArray)


def trendDataTest(timeSteps):
    dampening = 2.0 * float(1.0 / timeSteps)
    power = 0.35
    displacement = -2.5

    f1 = gT.linearTrendFn
    data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

    f2 = gT.logTrendFn
    data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    f3 = gT.negExpTrendFn
    t3 = gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    # plt.plot(t2)
    # plt.show()

    return data


def harmonicDataTest(timeSteps):
    sineCoeffs = [-2.0, 3.0]
    sinePeriods = [26.0, 30.0]

    cosineCoeffs = [-2.5]
    cosinePeriods = [16.0]

    data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
    # plt.plot(data)
    # plt.show()

    return data


# test for a single time series imputation and forecasting
def testSingleTS():
    print("------------------- Test # 1 (Single TS). ------------------------")
    p = 0.7
    N = 50
    M = 400
    timeSteps = N * M

    # train/test split
    trainProp = 0.9
    M1 = int(trainProp * M)
    M2 = M - M1

    trainPoints = N * M1
    testPoints = N * M2

    print("Generating data...")
    harmonicsTS = harmonicDataTest(timeSteps)
    trendTS = trendDataTest(timeSteps)
    (armaTS, armaMeanTS) = armaDataTest(timeSteps)

    meanTS = harmonicsTS + trendTS + armaMeanTS
    combinedTS = harmonicsTS + trendTS + armaTS

    # normalize the values to all lie within [-1, 1] -- helps with RMSE comparisons
    # can use the tsUtils.unnormalize() function to convert everything back to the original range at the end, if needed
    max1 = np.nanmax(combinedTS)
    min1 = np.nanmin(combinedTS)
    max2 = np.nanmax(meanTS)
    min2 = np.nanmin(meanTS)
    max = np.max([max1, max2])
    min = np.min([min1, min2])

    combinedTS = tsUtils.normalize(combinedTS, max, min)
    meanTS = tsUtils.normalize(meanTS, max, min)

    # produce timestamps
    timestamps = np.arange('2017-09-10 20:30:00', timeSteps, dtype='datetime64[1m]')  # arbitrary start date

    # split the data
    trainDataMaster = combinedTS[0:trainPoints]  # need this as the true realized values for comparisons later
    meanTrainData = meanTS[0:trainPoints]  # this is only needed for various statistical comparisons later

    # randomly hide training data: choose between randomly hiding entries or randomly hiding consecutive entries
    (trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster), p)

    # now further hide consecutive entries for a very small fraction of entries in the eventual training matrix
    (trainData, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData), 0.9, int(M1 * 0.25),
                                                                       M1)

    # interpolating Nans with linear interpolation
    # trainData = tsUtils.nanInterpolateHelper(trainData)

    # test data and hidden truth
    testData = combinedTS[-1 * testPoints:]
    meanTestData = meanTS[-1 * testPoints:]  # this is only needed for various statistical comparisons

    # time stamps
    trainTimestamps = timestamps[0:trainPoints]
    testTimestamps = timestamps[-1 * testPoints:]

    # once we have interpolated, pObservation should be set back to 1.0
    pObservation = 1.0

    # create pandas df
    key1 = 't1'
    trainMasterDF = pd.DataFrame(index=trainTimestamps, data={key1: trainDataMaster})  # needed for reference later
    trainDF = pd.DataFrame(index=trainTimestamps, data={key1: trainData})
    meanTrainDF = pd.DataFrame(index=trainTimestamps, data={key1: meanTrainData})

    testDF = pd.DataFrame(index=testTimestamps, data={key1: testData})
    meanTestDF = pd.DataFrame(index=testTimestamps, data={key1: meanTestData})

    # train the model
    print("Training the model (imputing)...")
    print('SVD')
    nbrSingValuesToKeep = 5
    mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy',
                   otherSeriesKeysArray=[], includePastDataOnly=True)
    mod.fit(trainDF)
    imputedDf = mod.denoisedDF()

    print(" RMSE (training imputation vs mean) = %f" % tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
    print(" RMSE (training imputation vs obs)  = %f" % tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))
    return
    print('ALS')
    # uncomment below to run the ALS algorithm ; comment out the above line
    mod = ALSModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, otherSeriesKeysArray=[],
                   includePastDataOnly=True)
    mod.fit(trainDF)

    # imputed + denoised data 
    imputedDf = mod.denoisedDF()

    print(" RMSE (training imputation vs mean) = %f" % tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
    print(" RMSE (training imputation vs obs)  = %f" % tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

    print("Forecasting (#points = %d)..." % len(testDF))
    # test data is used for point-predictions
    forecastArray = []
    for i in range(0, len(testDF)):
        pastPoints = np.zeros(N - 1)  # need an N-1 length vector of past point
        j = 0
        if (i < N - 1):  # the first prediction uses the end of the training data
            while (j < N - 1 - i):
                pastPoints[j] = trainMasterDF[key1].values[len(trainDF) - (N - 1 - i) + j]
                j += 1

        if (j < N - 1):  # use the new test data
            pastPoints[j:] = testDF[key1].values[i - (N - 1) + j:i]

        keyToSeriesDFNew = pd.DataFrame(data={key1: pastPoints})
        prediction = mod.predict(pd.DataFrame(data={}), keyToSeriesDFNew, bypassChecks=False)
        forecastArray.append(prediction)

    print(" RMSE (prediction vs mean) = %f" % tsUtils.rmse(meanTestDF[key1].values, forecastArray))
    print(" RMSE (prediction vs obs)  = %f" % tsUtils.rmse(testDF[key1].values, forecastArray))

    print("Plotting...")
    plt.plot(np.concatenate((trainMasterDF[key1].values, testDF[key1].values), axis=0), color='gray', label='Observed')
    plt.plot(np.concatenate((meanTrainDF[key1].values, meanTestDF[key1].values), axis=0), color='red',
             label='True Means')
    plt.plot(np.concatenate((imputedDf[key1].values, forecastArray), axis=0), color='blue', label='Forecasts')
    plt.axvline(x=len(trainDF), linewidth=1, color='black', label='Training End')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.title('Single Time Series (ARMA + Periodic + Trend) - $p = %.2f$' % p)
    plt.show()


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    testSingleTS()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")


if __name__ == "__main__":
    main()
