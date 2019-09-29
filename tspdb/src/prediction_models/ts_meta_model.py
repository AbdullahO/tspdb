import numpy as np
import pandas as pd
from  tspdb.src.prediction_models.ts_svd_model import SVDModel
from math import ceil

class TSMM(object):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will not be trained.
    # T:                        (int) Number of entries in each submodel
    # gamma:                    (float) (0,1) fraction of T after which the model is updated
    # col_to_row_ratio:         (int) the ration of no. columns to the number of rows in each sub-model

    def __init__(self, kSingularValuesToKeep=3, T=int(1e5), gamma=0.2, T0=1000, col_to_row_ratio=1, SSVT=False, p=1.0, L=None, model_table_name='', persist_L = False):
        self.kSingularValuesToKeep = kSingularValuesToKeep
        
        if L is None:
            self.L = int(np.sqrt(T / col_to_row_ratio))
            M = int(self.L * col_to_row_ratio)
            self.T = int(self.L * M)
            self.col_to_row_ratio = col_to_row_ratio
      
        else:
            
            self.L = L
            M = int(T/L)
            self.T = int(self.L*M)
            self.col_to_row_ratio = M/L
      
        if M % 2 != 0:
            M = M+ 1
            self.T = self.L * M
            print ('Number of columns has to be even, thus T is changed into ', self.T)
        
        self.persist_L = persist_L
        self.gamma = gamma
        self.models = {}
        self.T0 = T0
        self.TimeSeries = None
        self.TimeSeriesIndex = 0
        self.ReconIndex = 0
        self.MUpdateIndex = 0
        self.model_tables_name = model_table_name
        self.SSVT = SSVT
        self.p = p

    def get_model_index(self, ts_index=None):
        if ts_index is None:
            ts_index = self.TimeSeriesIndex
        model_index = int(max((ts_index - 1) / (self.T / 2) - 1, 0))
        return model_index

    def update_model(self, NewEntries):
        """
        This function takes a new set of entries and update the model accordingly.
        if the number of new entries means new model need to be bulit, this function segment the new entries into
        several entries and then feed them to the update_ts and fit function
        :param NewEntries: Entries to be included in the new model
        """
        # Define update chunck for the update SVD function (Not really needed, should be resolved once the update function is fixed)

        if len(self.models) == 1 and len(NewEntries) < self.T / 2:
            UpdateChunk = 20 * int(np.sqrt(self.T0))
        else:
            UpdateChunk = int(self.T / (2 * (1+self.col_to_row_ratio) * 0.85))

        # find if new models should be constructed
        N = len(NewEntries)
        if N == 0:
            return

        current_no_models = len(self.models)
        updated_no_models = self.get_model_index(self.TimeSeriesIndex + N) + 1

        # if no new models are to be constructed
        if current_no_models == updated_no_models:
            # If it is a big update, do it at once
            last_model_size = self.models[updated_no_models - 1].M * self.models[updated_no_models - 1].N

            if len(NewEntries) / float(last_model_size) > self.gamma:
               
                self.updateTS(NewEntries[:])
                self.fitModels()
                return
            # Otherwise, update it chunk by chunk (Because Incremental Update requires small updates (not really, need to be fixed))
            i = -1
            for i in range(int(ceil(len(NewEntries)/(UpdateChunk)))):
                self.updateTS(NewEntries[i * UpdateChunk: (i + 1) * UpdateChunk])
                self.fitModels()

        else:
            # first complete the last model so it would have exactly T entries
            if current_no_models > 0:
                fillFactor = (self.TimeSeriesIndex % int(self.T / 2))
                FillElements = int((self.T / 2 - fillFactor)) * (fillFactor > 0)
                if FillElements > 0:
                    self.updateTS(NewEntries[:FillElements])
                    self.fitModels()
                    NewEntries = NewEntries[FillElements:]
            # Then, build the other new models. one of the is the very first model, we will skip the second iteration.
            SkipNext = False
            for i in range(updated_no_models - current_no_models + (current_no_models == 0)):
                if SkipNext:
                    SkipNext = False
                    continue
                if len(self.models) == 0:
                    self.updateTS(NewEntries[: self.T])
                    SkipNext = True
                    self.fitModels()
                    i += 1
                else:
                    self.updateTS(NewEntries[i * int(self.T / 2): (i + 1) * int(self.T / 2)])
                    self.fitModels()

    def updateTS(self, NewEntries):
        # Update the time series with the new entries.
        # only keep the last T entries

        n = len(NewEntries)

        if n > self.T / 2 and len(self.models) > 1:
            raise Exception('TimeSeries should be updated before T/2 values are assigned')

        self.TimeSeriesIndex += n

        if self.TimeSeriesIndex == n:
            self.TimeSeries = NewEntries

        elif len(self.TimeSeries) < self.T:
            TSarray = np.zeros(len(self.TimeSeries) + n)
            TSarray[:len(self.TimeSeries)] = self.TimeSeries
            TSarray[len(self.TimeSeries):] = NewEntries
            self.TimeSeries = TSarray

        else:
            if n < self.T: self.TimeSeries[:self.T - n] = self.TimeSeries[-self.T + n:]

            self.TimeSeries[-n:] = NewEntries

        if len(self.TimeSeries) > self.T:
            self.TimeSeries = self.TimeSeries[-self.T:]

    def fitModels(self):

        # Determine which model to fit
        ModelIndex = self.get_model_index(self.TimeSeriesIndex)
        # Determine the number of new Entries since the last reconstruction of a model
        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
        lenEntriesSinceLastUpdate = self.TimeSeriesIndex - self.MUpdateIndex
        # Do not fit very few observations
        if self.TimeSeriesIndex < self.T0:
            return
        # Do not fit a lot of observations
        if lenEntriesSinceLastUpdate > self.T and ModelIndex != 0:
            print (self.TimeSeriesIndex, self.MUpdateIndex, [(m.N, m.M, m.start) for m in self.models.values()])
            raise Exception('Model should be updated before T values are assigned')

        if lenEntriesSinceLastUpdate == 0:
            raise Exception('There are no new entries')

        # Build a new model
        if ModelIndex not in self.models:
            # start with the last T/2 entries from previous model
            initEntries = self.TimeSeries[int(int(self.T / 2) - self.TimeSeriesIndex % (self.T / 2)):]
            start = self.TimeSeriesIndex - self.TimeSeriesIndex % int(self.T / 2) - int(self.T / 2)
            # if ModelIndex != 0: assert len(initEntries) == self.T / 2
            rect = 1
            if lenEntriesSinceCons == self.T / 2 or ModelIndex == 0:
                initEntries = self.TimeSeries[:]
                start = max(self.TimeSeriesIndex - self.T, 0)

            if self.persist_L: N = self.L
            else: N = int(np.sqrt(len(initEntries) / (self.col_to_row_ratio)))
            M = int(len(initEntries) / N)
            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=int(start), SSVT=self.SSVT,
                                               probObservation=self.p)
            
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': initEntries[:int(N * M)]}))
            self.ReconIndex = N * M + start
            self.MUpdateIndex = self.ReconIndex

            if lenEntriesSinceCons == self.T / 2 or ModelIndex == 0:
                return
        Model = self.models[ModelIndex]

        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
        ModelLength = Model.N * Model.M + Model.start
        if (float(lenEntriesSinceCons) / (self.ReconIndex - Model.start) >= self.gamma) or (
                        self.TimeSeriesIndex % (self.T / 2) == 0):  # condition to create new model

            TSlength = self.TimeSeriesIndex - Model.start
            if self.persist_L: N = self.L
            else: N = int(np.sqrt(TSlength/self.col_to_row_ratio))
            M = int(TSlength / N)
            TSeries = self.TimeSeries[-TSlength:]
            TSeries = TSeries[:N * M]

            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start= int(Model.start),
                                               TimesReconstructed=Model.TimesReconstructed + 1,
                                               TimesUpdated=Model.TimesUpdated, SSVT=self.SSVT, probObservation=self.p)
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': TSeries}))
            self.ReconIndex = N * M + Model.start
            self.MUpdateIndex = self.ReconIndex

        else:

            Model = self.models[ModelIndex]
            N = Model.N
            if self.TimeSeriesIndex - ModelLength < N:
                pass
            else:
                D = self.TimeSeries[-(self.TimeSeriesIndex - ModelLength):]
                p = int(len(D) / N)

                D = D[:N * p]
                Model.updateSVD(D, 'UP')
                self.MUpdateIndex = Model.N * Model.M + Model.start
                Model.updated = True
                

    def _denoiseTS(self, models=None, index=None, range_=True):
        if models is None:
            models = self.models
        if range_ or index is None:
            if index is None:
                index = [0, int(self.MUpdateIndex)]
            denoised = np.zeros(index[1] - index[0])
            count = np.zeros(index[1] - index[0])
            y1, y2 = index[0], index[1]
            for Model in models.values():
                x1, x2 = Model.start, Model.M * Model.N + Model.start
                if x1 <= y2 and y1 <= x2:
                    RIndex = np.array([max(x1, y1), min(x2, y2)])
                    RIndexS = RIndex - y1
                    denoised[RIndexS[0]:RIndexS[1]] += Model.denoisedTS(RIndex - x1, range_)
                    count[RIndexS[0]:RIndexS[1]] += 1
            denoised[count == 0] = np.nan
            denoised[count > 0] = denoised[count > 0] / count[count > 0]
            return denoised

        else:

            index = np.array(index)
            I = len(index)
            denoised = np.zeros(I)
            models = np.zeros(2 * I)
            models[:I] = (index) / (self.T / 2) - 1
            models[I:] = models[:I] + 1
            models[models < 0] = 0
            count = np.zeros(len(index))

            for ModelNumber in np.unique(models):
                Model = models[ModelNumber]
                x1, x2 = Model.start, Model.M * Model.N + Model.start
                updatedIndices = np.logical_and(index >= x1, index < x2)
                assert np.sum(updatedIndices) > 0
                count += updatedIndices
                denoised[updatedIndices] += Model.denoisedTS(index[updatedIndices] - x1, range_)

            denoised[count == 0] = np.nan
            denoised[count > 0] = denoised[count > 0] / count[count > 0]
            return denoised

    def _predict(self, index=None, method='average', NoModels=None, dataPoints=None, models=None):

        if models is None:
            models = self.models
        n = len(models)

        if NoModels is None or NoModels > n or NoModels < 1: NoModels = n
        # if index next predict
        # UsedModels = [a for a in models.values()[-NoModels:]]
        UsedModels = [models[i] for i in range(n - NoModels, n)]

        if dataPoints is None and (index is None or index == self.TimeSeriesIndex + 1):
            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L:]})

            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is None and index <= self.TimeSeriesIndex:
            slack = self.TimeSeriesIndex - index + 1
            if slack > (self.T - self.L): raise Exception
            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L - slack:-slack]})
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is not None:
            assert len(dataPoints) == self.L - 1
            TSDF = pd.DataFrame(data={'t1': dataPoints})
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)
        else:
            return 0
