################################################################
#
# MultiDimensional Robust Synthetic Control (mRSC)
#
# Implementation based on: 
# url forthcoming (Paper to appear in ACM Sigmetrics 2019)
# (http://dna-pubs.cs.columbia.edu/citation/paperfile/230/mRSC.pdf)
#
################################################################
import numpy as np
import pandas as pd

from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src import tsUtils

class MultiRobustSyntheticControl(RobustSyntheticControl):
    
    # nbrMetrics:               (int) the number of metrics of interest
    # weightsArray:             (array) weight to scale each metric by. length of array must be == nbrMetrics
    # seriesToPredictKey:       (string) the series of interest (key)
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # M:                        (int) the number of columns in the matrix for EACH metric
    #                                total matrix columns shall be (nbrMetrics * M)
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # modelType:                (string) SVD or ALS. Default is "SVD"
    # svdMethod:                (string) the SVD method to use (optional)
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict 

    def __init__(self, nbrMetrics, weightsArray, seriesToPredictKey, kSingularValuesToKeep, M, probObservation=1.0, modelType='svd', svdMethod='numpy', otherSeriesKeysArray=[]):

        # set mRSC specific parms
        self.nbrMetrics = nbrMetrics
        self.weightsArray = weightsArray
            
        if (len(self.weightsArray) != self.nbrMetrics):
            raise Exception('len(weightsArray) must equal self.nbrMetrics')

        self.combinedM = self.nbrMetrics * M

        # initialize the super class
        super(MultiRobustSyntheticControl, self).__init__(seriesToPredictKey, kSingularValuesToKeep, self.combinedM, probObservation, modelType, svdMethod, otherSeriesKeysArray)
        

    # helper method to combine metrics appropriately
    # arrayOfKeyToSeriesDF:     (array) contains a keyToSeriesDF for each metric.
    #                                   length of array must equal the self.nbrMetrics
    #                                   order of the keyToSeriesDF in the array must remain consistent in fit() and predict().
    # isForTraining:            (bool) True if used in training and False if used for Predictions
    def combineMetrics(self, arrayOfKeyToSeriesDF, isForTraining):

        if (len(arrayOfKeyToSeriesDF) != self.nbrMetrics):
            raise Exception('len(arrayOfKeyToSeriesDF) must equal self.nbrMetrics')  

        dataDict = {}
        if (isForTraining):
            allKeys = [self.seriesToPredictKey] + self.otherSeriesKeysArray
        else:
            allKeys = self.otherSeriesKeysArray
        
        # scaling by the specified weights
        for i in range(0, self.nbrMetrics):
            arrayOfKeyToSeriesDF[i] = arrayOfKeyToSeriesDF[i].multiply(np.sqrt(self.weightsArray[i]))

        for k in allKeys:
            dataArray = []
            #print(k, self.nbrMetrics)
            for mInd in range(0, self.nbrMetrics):
                dfData = arrayOfKeyToSeriesDF[mInd]
                dataArray = dataArray + list(dfData[k].values)

            dataDict.update({k: dataArray})

        return pd.DataFrame(data=dataDict)

    # arrayOfKeyToSeriesDF:     (array) contains a keyToSeriesDF for each metric.
    #                             length of array must equal the self.nbrMetrics
    #                             order of the keyToSeriesDF in the array must remain consistent in fit() and predict().
    #                             same order must be followed for the self.weightsArray
    #
    # Note that the keys provided in the constructor MUST all be present in each keyToSeriesDF
    # The values must be all numpy arrays of floats.
    def fit(self, arrayOfKeyToSeriesDF):
    	super(MultiRobustSyntheticControl, self).fit(self.combineMetrics(arrayOfKeyToSeriesDF, True))


    # arrayOfKeyToSeriesDFNew:   (array) contains a keyToSeriesDFNew (Pandas dataframe) for each metric.
    #                              length of array must equal the self.nbrMetrics
    #                              order of the keyToSeriesDFNew in the array must remain consistent in fit() and predict().
    #                              same order must be followed for the self.weightsArray
    #
    #                              each keyToSeriesDFNew needs to contain all keys provided in the model;
    #                              all series MUST be of length >= 1, 
    #                              If longer than 1, then the most recent point will be used (for each series)
    #
    # Returns an array of prediction arrays, one for each metric
    def predict(self, arrayOfKeyToSeriesDFNew):
    	allPredictions = super(MultiRobustSyntheticControl, self).predict(self.combineMetrics(arrayOfKeyToSeriesDFNew, False))
        predictionsArray = []
        singleMetricPredictionsLength = int((1.0/float(self.nbrMetrics)) * len(allPredictions))
        for i in range(0, self.nbrMetrics):
            predictions = allPredictions[i * singleMetricPredictionsLength: (i+1) * singleMetricPredictionsLength]
            predictionsArray.append(predictions)

        return predictionsArray
