######################################################
#
# The Time Series Model based on ALS
#
######################################################
import numpy as np
from tslib.src.algorithms.alsWrapper import ALSWrapper as ALS
from tslib.src.algorithms.svdWrapper import SVDWrapper as SVD
from tslib.src.models.tsSVDModel import SVDModel

class ALSModel(SVDModel):

    # seriesToPredictKey:       (string) the time series of interest (key)
    # kFactors:    				(int) number of factors (similar to the kSingularValues of the parent class)
    # N:                        (int) the number of rows of the matrix for each series
    # M:                        (int) the number of columns for the matrix for each series
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict 
    # includePastDataOnly:      (Boolean) defaults to True. If this is set to False, 
    #                               the time series in 'otherSeriesKeysArray' will include the latest data point.
    #                               Note: the time series of interest (seriesToPredictKey) will never include 
    #                               the latest data-points for prediction
    def __init__(self, seriesToPredictKey, kFactors, N, M, probObservation=1.0, otherSeriesKeysArray=[], includePastDataOnly=True):

        super(ALSModel, self).__init__(seriesToPredictKey, kFactors, N, M, probObservation=probObservation, svdMethod='numpy', otherSeriesKeysArray=otherSeriesKeysArray, includePastDataOnly=includePastDataOnly)

    # run a least-squares regression of the last row of self.matrix and all other rows of self.matrix
    # sets and returns the weights
    # DO NOT call directly
    def _computeWeights(self):
        
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
            print(eachTSRows)
            while (rowIndex < matrixDim1):
            	newMatrix[rowIndex: rowIndex + eachTSRows] = self.matrix[matrixInd: matrixInd +eachTSRows]

            	rowIndex += eachTSRows
            	matrixInd += self.N

        self.weights = np.dot(np.linalg.pinv(newMatrix).T, self.lastRowObservations.T)


	# keyToSeriesDictionary: (Pandas dataframe) a key-value Series (time series)
    # Same as the parent class (SVDModel)
    def fit(self, keyToSeriesDF):

        # assign data to class variables
        super(ALSModel, self)._assignData(keyToSeriesDF, missingValueFill=False)

        self.max = np.nanmax(self.matrix)
        self.min = np.nanmin(self.matrix)

        # now use ALS to produce an estimated matrix
        alsMod = ALS(self.matrix, method='als')
        (U, V) = alsMod.reconstructMatrix(self.kSingularValues, 0.0, returnMatrix=False, tol=1e-9)

        self.matrix = np.dot(U, V)

        self.matrix[self.matrix > self.max] = self.max
        self.matrix[self.matrix < self.min] = self.min

        # we need to assign some values to the lastRowObservations where there are still NaNs
        # impute those with the ALS-estimated/iputed values
        for i in range(0, len(self.lastRowObservations)):
        	if (np.isnan(self.lastRowObservations[i])):
        		self.lastRowObservations[i] = self.matrix[-1, i]

        # set weights (same as the parent class now that we have the SVD of the ALS-estimated matrix)
        self._computeWeights()
	
	# return the imputed matrix, same as the parent class (SVDModel)
    def denoisedDF(self):

    	return super(ALSModel, self).denoisedDF()

	# same params as the predict() method of the parent class (SVDModel)       
    def predict(self, otherKeysToSeriesDFNew, predictKeyToSeriesDFNew, bypassChecks=False):

    	return super(ALSModel, self).predict(otherKeysToSeriesDFNew, predictKeyToSeriesDFNew, bypassChecks)

    def updateSVD(self, D):
        return super(ALSModel, self).updateSVD(self, D)