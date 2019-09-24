################################################################
#
# Robust Synthetic Control 
#
# Implementation based on: 
# http://www.jmlr.org/papers/volume19/17-777/17-777.pdf
#
################################################################
import numpy as np
import pandas as pd

from tslib.src.models.tsSVDModel import SVDModel
from tslib.src.models.tsALSModel import ALSModel
from tslib.src import tsUtils

class RobustSyntheticControl(object):
    
    # seriesToPredictKey:       (string) the series of interest (key)
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # M:                        (int) the number of columns for the matrix
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # modelType:                (string) SVD or ALS. Default is "SVD"
    # svdMethod:                (string) the SVD method to use (optional)
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict 

    def __init__(self, seriesToPredictKey, kSingularValuesToKeep, M, probObservation=1.0, modelType='svd', svdMethod='numpy', otherSeriesKeysArray=[]):

        self.seriesToPredictKey = seriesToPredictKey
        self.otherSeriesKeysArray = otherSeriesKeysArray

        self.N = 1 # each series is on its own row
        self.M = M

        self.kSingularValues = kSingularValuesToKeep
        self.svdMethod = svdMethod

        self.p = probObservation

        if (modelType == 'als'):
            self.model = ALSModel(self.seriesToPredictKey, self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)

        elif (modelType == 'svd'):
            self.model = SVDModel(self.seriesToPredictKey, self.kSingularValues, self.N, self.M, probObservation=self.p, svdMethod='numpy', otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)

        else:
            self.model = SVDModel(self.seriesToPredictKey, self.kSingularValues, self.N, self.M, probObservation=self.p, svdMethod='numpy', otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)

        self.control = None # these are the synthetic control weights


    # keyToSeriesDictionary: (Pandas dataframe) a key-value Series
    # Note that the keys provided in the constructor MUST all be present
    # The values must be all numpy arrays of floats.
    def fit(self, keyToSeriesDF):

    	self.model.fit(keyToSeriesDF)


    # otherKeysToSeriesDFNew:     (Pandas dataframe) needs to contain all keys provided in the model;
    #                               all series/array MUST be of length >= 1, 
    #                               If longer than 1, then the most recent point will be used (for each series)
    def predict(self, otherKeysToSeriesDFNew):
        prediction = np.dot(self.model.weights, otherKeysToSeriesDFNew[self.otherSeriesKeysArray].T)
    	return prediction

    # return the synthetic control weights
    def getControl():

    	if (self.model.weights is None):
    		raise Exception('Before calling getControl() you need to call the fit() method first.')

    	else:
    		return self.model.weights
