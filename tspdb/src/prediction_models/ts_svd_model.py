######################################################
#
# The Time Series Model based on SVD
#
######################################################
import copy
import numpy as np
import pandas as pd
from tspdb.src.algorithms.svdWrapper import SVDWrapper as SVD
from tspdb.src import tsUtils
from sklearn.metrics import r2_score
class SVDModel(object):

    # seriesToPredictKey:       (string) the time series of interest (key)
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # N:                        (int) the number of rows of the matrix for each series
    # M:                        (int) the number of columns for the matrix for each series
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # svdMethod:                (string) the SVD method to use (optional)
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict 
    # includePastDataOnly:      (Boolean) defaults to True. If this is set to False, 
    #                               the time series in 'otherSeriesKeysArray' will include the latest data point.
    #                               Note: the time series of interest (seriesToPredictKey) will never include 
    #                               the latest data-points for prediction
    def __init__(self, seriesToPredictKey, kSingularValuesToKeep, N, M,updated = True, probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True, start = 0, TimesUpdated = 0, TimesReconstructed =0, SSVT = False ):

        self.seriesToPredictKey = seriesToPredictKey
        self.otherSeriesKeysArray = otherSeriesKeysArray
        self.includePastDataOnly = includePastDataOnly

        self.N = N
        self.M = M
        self.start = start
        self.TimesUpdated = TimesUpdated
        self.TimesReconstructed = TimesReconstructed
        self.kSingularValues = kSingularValuesToKeep
        self.svdMethod = svdMethod

        self.Uk = None
        self.Vk = None
        self.sk = None
        self.matrix = None
        self.lastRowObservations = None
        self.Ukw = None
        self.Vkw = None
        self.skw = None
        self.p = probObservation
        self.forecast_model_score = 0
        self.imputation_model_score = 0
        self.weights = None
        self.SSVT = SSVT
        self.soft_threshold = 0
        self.updated = updated
    # run a least-squares regression of the last row of self.matrix and all other rows of self.matrix
    # sets and returns the weights
    # DO NOT call directly
    def _computeWeights(self):       

        ### This is now the same as ALS
        ## this is an expensive step because we are computing the SVD all over again 
        ## however, currently, there is no way around it since this is NOT the same matrix as the full
        ## self.matrix, i.e. we have fewer (or just one less) rows

        if (self.lastRowObservations is None):
            raise Exception('Do not call _computeWeights() directly. It should only be accessed via class methods.')

        # need to decide how to produce weights based on whether the N'th data points are to be included for the other time series or not
        # for the seriesToPredictKey we only look at the past. For others, we could be looking at the current data point in time as well.
        
        matrixDim1 = (self.N * len(self.otherSeriesKeysArray)) + self.N-1
        matrixDim2 = np.shape(self.matrix)[1]
        eachTSRows = self.N

        if (self.includePastDataOnly == False):
            newMatrix = self.matrix[0:matrixDim1, :]

        else:
            matrixDim1 = ((self.N - 1) * len(self.otherSeriesKeysArray)) + self.N-1
            eachTSRows = self.N - 1

            newMatrix = np.zeros([matrixDim1, matrixDim2])

            rowIndex = 0
            matrixInd = 0

            while (rowIndex < matrixDim1):
                newMatrix[rowIndex: rowIndex + eachTSRows] = self.matrix[matrixInd: matrixInd +eachTSRows]

                rowIndex += eachTSRows
                matrixInd += self.N

        svdMod = SVD(newMatrix, method='numpy')
        (self.skw, self.Ukw, self.Vkw) = svdMod.reconstructMatrix(self.kSingularValues, returnMatrix=False)
        soft_threshold = 0
        if self.SSVT: soft_threshold = svdMod.next_sigma
        matrix = tsUtils.matrixFromSVD(self.skw, self.Ukw, self.Vkw, soft_threshold=soft_threshold)
        newMatrixPInv = tsUtils.pInverseMatrixFromSVD(self.skw, self.Ukw, self.Vkw,soft_threshold=soft_threshold)
        self.weights = np.dot(newMatrixPInv.T, self.matrix[-1,:].T)
        self.forecast_model_score = r2_score( self.lastRowObservations, np.dot(matrix.T,self.weights))

    # return the imputed matrix
    def denoisedDF(self):
        setAllKeys = set(self.otherSeriesKeysArray)
        setAllKeys.add(self.seriesToPredictKey)

        single_ts_rows = self.N
        dataDict = {}
        rowIndex = 0
        for key in self.otherSeriesKeysArray:

            dataDict.update({key: self.matrix[rowIndex*single_ts_rows: (rowIndex+1)*single_ts_rows, :].flatten('F')})
            rowIndex += 1

        dataDict.update({self.seriesToPredictKey: self.matrix[rowIndex*single_ts_rows: (rowIndex+1)*single_ts_rows, :].flatten('F')})

        return pd.DataFrame(data=dataDict)


    def denoisedTS(self, ind = None, range = True):
        if self.matrix is None:
            self.matrix =  tsUtils.matrixFromSVD(self.sk, self.Uk, self.Vk, self.soft_threshold,probability=self.p)
        NewColsDenoised = self.matrix.flatten('F')
        if ind is None:
            return NewColsDenoised
        if range:
            assert len(ind) == 2
            return NewColsDenoised[ind[0]:ind[1]]
        else:
            return NewColsDenoised[ind]

    def _assignData(self, keyToSeriesDF, missingValueFill=True):

        setAllKeys = set(self.otherSeriesKeysArray)
        setAllKeys.add(self.seriesToPredictKey)

        if (len(set(keyToSeriesDF.columns.values).intersection(setAllKeys)) != len(setAllKeys)):
            raise Exception('keyToSeriesDF does not contain ALL keys provided in the constructor.')

        if (missingValueFill == True):
            # impute with the least informative value (middle)
            # max = np.nanmax(keyToSeriesDF)

            # min = np.nanmin(keyToSeriesDF)
            # diff = 0.5*(min + max)
            keyToSeriesDF = keyToSeriesDF.fillna(method = 'ffill')
        else:
            keyToSeriesDF = keyToSeriesDF.fillna(value = 0)
        T = self.N * self.M
        for key in setAllKeys:
            if (len(keyToSeriesDF[key]) < T):
                raise Exception('All series (columns) provided must have length >= %d' %T)


        # initialize the matrix of interest
        single_ts_rows = self.N
        matrix_cols = self.M
        matrix_rows = int(len(setAllKeys) * single_ts_rows)
  
        self.matrix = np.zeros([matrix_rows, matrix_cols])

        seriesIndex = 0
        for key in self.otherSeriesKeysArray: # it is important to use the order of keys set in the model
            self.matrix[seriesIndex*single_ts_rows: (seriesIndex+1)*single_ts_rows, :] = tsUtils.arrayToMatrix(keyToSeriesDF[key][-1*T:].values, single_ts_rows, matrix_cols)
            seriesIndex += 1

        # finally add the series of interest at the bottom
       # tempMatrix = tsUtils.arrayToMatrix(keyToSeriesDF[self.seriesToPredictKey][-1*T:].values, self.N, matrix_cols)
        self.matrix[seriesIndex*single_ts_rows: (seriesIndex+1)*single_ts_rows, :] = tsUtils.arrayToMatrix(keyToSeriesDF[self.seriesToPredictKey][-1*T:].values, single_ts_rows, matrix_cols)
        
        # set the last row of observations
        self.lastRowObservations = copy.deepcopy(self.matrix[-1, :])


    # keyToSeriesDictionary: (Pandas dataframe) a key-value Series (time series)
    # Note that the keys provided in the constructor MUST all be present
    # The values must be all numpy arrays of floats.
    # This function sets the "de-noised" and imputed data matrix which can be accessed by the .matrix property
    def fit(self, keyToSeriesDF):

        # assign data to class variables
        obs = keyToSeriesDF[self.seriesToPredictKey].values
        filter_nan = ~np.isnan(obs)
        obs = obs[filter_nan]
        self._assignData(keyToSeriesDF, missingValueFill=True)
        # now produce a thresholdedthresholded/de-noised matrix. this will over-write the original data matrix
        svdMod = SVD(self.matrix, method='numpy')
        (self.sk, self.Uk, self.Vk) = svdMod.reconstructMatrix(self.kSingularValues, returnMatrix=False)
        if self.SSVT: self.soft_threshold = svdMod.next_sigma
        # set weights
        self.matrix = tsUtils.matrixFromSVD(self.sk, self.Uk, self.Vk, self.soft_threshold,probability=self.p)
        self.imputation_model_score = r2_score(obs,self.denoisedTS()[filter_nan])
        self._computeWeights()

    def updateSVD(self,D, method = 'UP', missingValueFill = True):
        assert (len(D) % self.N == 0)
        if (missingValueFill == True):
            # impute with the least informative value (middle)
            max = np.nanmax(D)
            if np.isnan(max): max = 0
            min = np.nanmin(D)
            if np.isnan(min): min = 0
            diff = 0.5*(min + max)
            D[np.isnan(D)] = diff
        else: D[np.isnan(D)] = 0
        D = D.reshape([self.N,int(len(D)/self.N)])

        assert D.shape[0] == self.N
        assert D.shape[1] <= D.shape[0]
       
        if method == 'UP':
            self.Uk, self.sk, self.Vk = tsUtils.updateSVD2(D, self.Uk, self.sk, self.Vk)
            self.M = self.Vk.shape[0]
            self.Ukw, self.skw, self.Vkw = tsUtils.updateSVD2(D[:-1,:], self.Ukw, self.skw, self.Vkw)

        elif method == 'folding-in':
            self.Uk, self.sk, self.Vk = tsUtils.updateSVD(D, self.Uk, self.sk ,self.Vk )

            self.M = self.Vk.shape[0]
            self.Ukw, self.skw, self.Vkw = tsUtils.updateSVD(D[:-1, :], self.Ukw, self.skw, self.Vkw)
        # elif method == 'Full':
        #     raise ValueError
        #     self.matrix = np.concatenate((self.matrix,D),1)
        #     U, S, V = np.linalg.svd(self.matrix, full_matrices=False)
        #     self.sk = S[0:self.kSingularValues]
        #     self.Uk = U[:, 0:self.kSingularValues]
        #     self.Vk = V[0:self.kSingularValues,:]
        #     self.Vk = self.Vk.T
        #     self.M = self.Vk.shape[0]
        else:
            raise ValueError
        
        self.matrix = tsUtils.matrixFromSVD(self.sk, self.Uk, self.Vk, self.soft_threshold,probability=self.p)
        self.lastRowObservations = self.matrix[-1,:]
        self.TimesUpdated +=1
        newMatrixPInv = tsUtils.pInverseMatrixFromSVD(self.skw, self.Ukw, self.Vkw,soft_threshold=self.soft_threshold, probability=self.p)
        self.weights = np.dot(newMatrixPInv.T, self.lastRowObservations.T)
        


    # otherKeysToSeriesDFNew:     (Pandas dataframe) needs to contain all keys provided in the model;
    #                           If includePastDataOnly was set to True (default) in the model, then:
    #                               each series/array MUST be of length >= self.N - 1
    #                               If longer than self.N - 1, then the most recent self.N - 1 points will be used
    #                           If includePastDataOnly was set to False in the model, then:
    #                               all series/array except seriesToPredictKey MUST be of length >= self.N (i.e. includes the current), 
    #                               If longer than self.N, then the most recent self.N points will be used
    #
    # predictKeyToSeriesDFNew:   (Pandas dataframe) needs to contain the seriesToPredictKey and self.N - 1 points past points.
    #                           If more points are provided, the most recent self.N - 1 points are selected.   
    #
    # bypassChecks:         (Boolean) if this is set to True, then it is the callee's responsibility to provide
    #                           all required series of appropriate lengths (see above).
    #                           It is advised to leave this set to False (default).         
    def predict(self, otherKeysToSeriesDFNew, predictKeyToSeriesDFNew, bypassChecks=False):

        nbrPointsNeeded = self.N - 1
        if (self.includePastDataOnly == False):
            nbrPointsNeeded = self.N

        if (bypassChecks == False):

            if (self.weights is None):
                raise Exception('Before predict() you need to call "fit()" on the model.')

            if (len(set(otherKeysToSeriesDFNew.columns.values).intersection(set(self.otherSeriesKeysArray))) < len(set(self.otherSeriesKeysArray))):
                raise Exception('keyToSeriesDFNew does not contain ALL keys provided in the constructor.')

            for key in self.otherSeriesKeysArray:
                points = len(otherKeysToSeriesDFNew[key])
                if (points < nbrPointsNeeded):
                    raise Exception('Series (%s) must have length >= %d' %(key, nbrPointsNeeded))

            points = len(predictKeyToSeriesDFNew[self.seriesToPredictKey])
            if (points < self.N - 1):
                raise Exception('Series (%s) must have length >= %d' %(self.seriesToPredictKey, self.N - 1))

        newDataArray = np.zeros((len(self.otherSeriesKeysArray) * nbrPointsNeeded) + self.N - 1)
        indexArray = 0
        for key in self.otherSeriesKeysArray:
            newDataArray[indexArray: indexArray + nbrPointsNeeded] = otherKeysToSeriesDFNew[key][-1*nbrPointsNeeded: ].values

            indexArray += nbrPointsNeeded

        # at last fill in the time series of interest
        newDataArray[indexArray:] = predictKeyToSeriesDFNew[self.seriesToPredictKey][-1*(self.N - 1):].values

        # dot product
        # newDataArray[np.isnan(newDataArray)] = 0
        projection = newDataArray#np.dot(self.Ukw, np.dot(newDataArray, self.Ukw).T)
        return np.dot(self.weights, projection)


