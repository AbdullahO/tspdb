import numpy as np
import pandas as pd
from  tspdb.src.prediction_models.ts_svd_model import SVDModel
from math import ceil
from sklearn.preprocessing import StandardScaler

class TSMM(object):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will not be trained.
    # T:                        (int) Number of entries in each submodel
    # gamma:                    (float) (0,1) fraction of T after which the model is updated
    # col_to_row_ratio:         (int) the ration of no. columns to the number of rows in each sub-model

    def __init__(self, kSingularValuesToKeep=None, T=int(1e5), gamma=0.2, T0=1000, col_to_row_ratio=1, SSVT=False, p=None, L=None, model_table_name='', persist_L = False, no_ts = 1, normalize = True, fill_in_missing = True):
        self.kSingularValuesToKeep = kSingularValuesToKeep
        
        self.no_ts = no_ts
        self.col_to_row_ratio = col_to_row_ratio
        self.fill_in_missing = fill_in_missing 
        # if self.col_to_row_ratio % (self.no_ts*2) != 0:
        #     self.col_to_row_ratio = self.col_to_row_ratio + (2*self.no_ts-self.col_to_row_ratio %(2*self.no_ts))
        # print(self.col_to_row_ratio)
        if L is None:
            self.L = int(np.sqrt(T / self.col_to_row_ratio))
            M = int(self.L * self.col_to_row_ratio)
            self.T = int(self.L * M)

        else:
            self.L = L
            M = int(T/L)
            self.T = int(self.L*M)
            self.col_to_row_ratio = M/L
        
        if M % (2*self.no_ts) != 0:
            M = M + (2*self.no_ts -M %(2*self.no_ts))
            # self.col_to_row_ratio = self.get_dimensions(M)
            # if not persist_L: self.L = self.T//M 
            # subtract a small amount to avoid issues with machine precision
            self.col_to_row_ratio = M/self.L -1e-14
            
            if not persist_L:
                self.T = int(M*M/self.col_to_row_ratio)
                self.L = int(np.sqrt(self.T / self.col_to_row_ratio))
            else:
                self.T = self.L*M
            print ('Number of columns has to be even and divisible by the number of time series, thus T is changed into %s, and col_to_row_ratio to %s'%(self.T, self.col_to_row_ratio))
  
        self.normalize = normalize
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

    def get_dimensions(self, M):
        ratio = self.col_to_row_ratio
        while M%ratio !=0:
            ratio +=1
        return ratio

    def get_model_index(self, ts_index=None):
        if ts_index is None:
            ts_index = self.TimeSeriesIndex
        model_index = int(max((ts_index - 1) / (self.T / 2) - 1, 0))
        return model_index

    def update_model(self, NewEntries):
        """
        This function takes a new se
        t of entries and update the model accordingly.
        if the number of new entries means new model need to be bulit, this function segment the new entries into
        several entries and then feed them to the update_ts and fit function
        :param NewEntries: Entries to be included in the new model
        """
        assert len(NewEntries.shape) == 2
        assert NewEntries.shape[1] == self.no_ts

        # Define update chunck for the update SVD function (Not really needed, should be resolved once the update function is fixed)
        if len(self.models) == 1 and NewEntries.size < self.T / 2:
            UpdateChunk = 20 * int(np.sqrt(self.T0))//self.no_ts
        else:
            UpdateChunk = int(self.T/(4*self.col_to_row_ratio))//self.no_ts
  

        # find if new models should be constructed
        N = NewEntries.size
        Rows = NewEntries.shape[0]
        
        if N == 0:
            return
        # if the number of models is zero, get the estimate of p
        if len(self.models) == 0 and self.p == None:
            self.p = 1.0 - np.sum(np.isnan(NewEntries))/NewEntries.size
            if self.fill_in_missing: self.p = 1.0
        current_no_models = len(self.models)
        updated_no_models = self.get_model_index(self.TimeSeriesIndex + N) + 1

        # if no new models are to be constructed
        if current_no_models == updated_no_models:
            print('same models')
            # If it is a big update, do it at once
            last_model_size = self.models[updated_no_models - 1].M * self.models[updated_no_models - 1].N

            if N/float(last_model_size) > self.gamma: # or len(self.models)>1:
                self.updateTS(NewEntries[:,:])
                self.fitModels()
                return
            # Otherwise, update it chunk by chunk (Because Incremental Update requires small updates when we have one sub-model)
            i = -1
            for i in range(int(ceil(Rows/(UpdateChunk)))):
                self.updateTS(NewEntries[i * UpdateChunk: (i + 1) * UpdateChunk,:])
                self.fitModels()

        else:
            # first complete the last model so it would have exactly T entries
            if current_no_models > 0:
                fillFactor = (self.TimeSeriesIndex % int(self.T /2))
                FillElements = (int((self.T / 2 - fillFactor)) * (fillFactor > 0)) // self.no_ts
                print('Fill', FillElements)
                if FillElements > 0:
                    self.updateTS(NewEntries[:FillElements,:])
                    self.fitModels()
                    NewEntries = NewEntries[FillElements:,:]
            # Then, build the other new models. one of the is the very first model, we will skip the second iteration.
            SkipNext = False
            
            for i in range(updated_no_models - current_no_models + (current_no_models == 0)):
                if SkipNext:
                    SkipNext = False
                    continue
                
                if len(self.models) == 0:
                    self.updateTS(NewEntries[: self.T//self.no_ts,:])
                    SkipNext = True
                    self.fitModels()
                    i += 1
                    if 0 in self.models:
                        self.kSingularValuesToKeep = self.models[0].kSingularValues

                else:
                    self.updateTS(NewEntries[i * int((self.T//self.no_ts)/2): (i + 1) * int((self.T//self.no_ts)/ 2),:])
                    self.fitModels()

    def updateTS(self, NewEntries):
        # Update the time series with the new entries.
        # only keep the last T entries

        N = NewEntries.size
        num_ts_obs = self.T//self.no_ts
        num_new_rows = N//self.no_ts

        if N > self.T / 2 and len(self.models) > 1:
            raise Exception('TimeSeries should be updated before T/2 values are assigned')

        self.TimeSeriesIndex += N

        if self.TimeSeriesIndex == N or self.TimeSeries is None:
            self.TimeSeries = NewEntries

        elif self.TimeSeries.size < self.T:
            TSarray = np.zeros([len(self.TimeSeries) + N//self.no_ts, self.no_ts])
            TSarray[:len(self.TimeSeries),:] = self.TimeSeries
            TSarray[len(self.TimeSeries):,:] = NewEntries
            self.TimeSeries = TSarray

        else:
            self.TimeSeries[:num_ts_obs-num_new_rows,:] = self.TimeSeries[-num_ts_obs + num_new_rows:,:]
            self.TimeSeries[-num_new_rows:,:] = NewEntries

        if self.TimeSeries.shape[0] > num_ts_obs:

            self.TimeSeries = self.TimeSeries[-num_ts_obs:,:]

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
            initEntries = self.TimeSeries[int(int(self.T / 2) - self.TimeSeriesIndex % (self.T / 2))//self.no_ts:,:]
            start = self.TimeSeriesIndex - self.TimeSeriesIndex % int(self.T / 2) - int(self.T / 2)
            # if ModelIndex != 0: assert len(initEntries) == self.T / 2
            rect = 1
            if lenEntriesSinceCons == self.T // 2:
                initEntries = self.TimeSeries[:,:]
                start = max(self.TimeSeriesIndex - self.T, 0)
            if ModelIndex == 0:
                initEntries = self.TimeSeries[:,:]
                start = 0

            if self.persist_L: N = self.L
            else: 
                N = int(np.sqrt(initEntries.size / (self.col_to_row_ratio)))
                if N >  initEntries.shape[0]:
                    N = initEntries.shape[0]
            M = int(initEntries.size / N)
            if M < self.no_ts:
                raise Exception ('Number of columns in the matrix (%s) is less than the number of time series (%s)' % (M, self.no_ts))
            if M%self.no_ts != 0:
                M -= M%self.no_ts

            M_ts = M//self.no_ts
            inc_obs = initEntries[:M_ts*N,:]
            
            if self.normalize:
                scaler = StandardScaler()
                inc_obs = scaler.fit_transform(inc_obs)
                norm_means = scaler.mean_
                norm_std = scaler.scale_
            else:
                norm_means = np.zeros(self.no_ts)
                norm_std = np.ones(self.no_ts)

            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=int(start), SSVT=self.SSVT,
                                               probObservation=self.p, no_ts = self.no_ts, norm_mean = norm_means, norm_std = norm_std, fill_in_missing = self.fill_in_missing)
            flattened_obs = inc_obs.reshape([N,M], order = 'F')
            flattened_obs = flattened_obs[:,np.arange(M_ts*self.no_ts).reshape([self.no_ts,M_ts]).flatten('F')]
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': flattened_obs.flatten('F')}))
            self.models[ModelIndex].obs_ = flattened_obs.flatten('F') 

            old_mupdate_index = self.MUpdateIndex
            self.ReconIndex = max(N * M + start, old_mupdate_index)
            self.MUpdateIndex = self.ReconIndex
            if lenEntriesSinceCons == self.T // 2 or ModelIndex == 0:
                return
        
        Model = self.models[ModelIndex]
        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
        ModelLength = Model.N * Model.M + Model.start
        TSlength = self.TimeSeriesIndex - Model.start
        if (TSlength <= self.TimeSeries.size and (float(lenEntriesSinceCons) / (self.ReconIndex - Model.start) >= self.gamma)) or (self.TimeSeriesIndex % (self.T / 2) == 0):  # condition to recompute SVD
            if self.persist_L: N = self.L
            else: N = int(np.sqrt(TSlength/self.col_to_row_ratio))
            M = int(TSlength / N)
            if M % self.no_ts != 0:
                M -= M%self.no_ts

            M_ts = M//self.no_ts
            TSeries = self.TimeSeries[-TSlength//self.no_ts:,:]
            TSeries = TSeries[:(N * M)//self.no_ts,:]
            if self.normalize:
                scaler = StandardScaler()
                TSeries = scaler.fit_transform(TSeries)
                norm_means = scaler.mean_
                norm_std = scaler.scale_
            else:
                norm_means = np.zeros(self.no_ts)
                norm_std = np.ones(self.no_ts)

            flattened_obs = TSeries.reshape([N,M], order = 'F')
            flattened_obs = flattened_obs[:,np.arange(M).reshape([self.no_ts,M_ts]).flatten('F')]
            
            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start= int(Model.start),
                                               TimesReconstructed=Model.TimesReconstructed + 1,
                                               TimesUpdated=Model.TimesUpdated, SSVT=self.SSVT, probObservation=self.p, 
                                               no_ts = self.no_ts, norm_mean = norm_means, norm_std = norm_std, fill_in_missing = self.fill_in_missing)
            
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': flattened_obs.flatten('F')}))
            self.ReconIndex = N * M + Model.start
            self.MUpdateIndex = self.ReconIndex

        else:
            Model = self.models[ModelIndex]
            N = Model.N
            M = Model.M
            if self.TimeSeriesIndex - ModelLength < N*self.no_ts:
                return
            else:
                NewEntries = self.TimeSeries[-(self.TimeSeriesIndex - ModelLength)//self.no_ts:,:]
                if self.normalize:
                    NewEntries = NewEntries - Model.norm_mean
                    NewEntries = NewEntries / Model.norm_std
                num_new_columns = self.no_ts*((NewEntries[:,0]).size//N)
                flattened_obs = NewEntries[:(num_new_columns*N)//self.no_ts,:].reshape([N,num_new_columns], order = 'F')
                flattened_obs = flattened_obs[:,np.arange(num_new_columns).reshape([self.no_ts,num_new_columns//self.no_ts]).flatten('F')]
                D = flattened_obs.flatten('F')
                # p = int(len(D) / N)
                # D = D[:N * p]
                print(D.shape)
                Model.updateSVD(D, 'UP')
                self.MUpdateIndex = Model.N * Model.M + Model.start
                Model.updated = True
                

    def _denoiseTS(self, models=None, index=None, range_=True):
        # denoise the whole time series if no specific  submodels are selected
        if models is None:
            models = self.models
        
        if range_ or index is None:
            if index is None:
                index = [0, int(self.MUpdateIndex)]
            no_obs = (index[1] - index[0])//self.no_ts
            denoised = np.zeros([no_obs, self.no_ts])
            count =  np.zeros([no_obs, self.no_ts])
            y1, y2 = index[0], index[1]
            for Model in models.values():
                x1, x2 = Model.start, Model.M * Model.N + Model.start
                if x1 <= y2 and y1 <= x2:
                    RIndex = np.array([max(x1, y1), min(x2, y2)])//self.no_ts
                    RIndexS = RIndex - y1//self.no_ts
                    # call to make sure that model.matrix is estimated
                    denoise_flat = Model.denoisedTS(return_ = False)
                    M = Model.M
                    N = Model.N
                    denoised_matrix = Model.matrix
                    denoised_columns_swapped = denoised_matrix[:,np.arange(M).reshape([M//self.no_ts, self.no_ts]).flatten('F')]
                    denoised_ts = denoised_columns_swapped.reshape([denoised_columns_swapped.size//self.no_ts,self.no_ts],order ='F')
                    denoised_index = RIndex-x1//self.no_ts
                    denoised[RIndexS[0]:RIndexS[1],:] += (denoised_ts[denoised_index[0]:denoised_index[1],:]*Model.norm_std+Model.norm_mean)
                    count[RIndexS[0]:RIndexS[1],:] += 1
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
