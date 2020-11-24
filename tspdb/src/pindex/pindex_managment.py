import numpy as np
import pandas as pd
from tspdb.src.database_module.db_class import Interface
from  tspdb.src.prediction_models.ts_meta_model import TSMM
from  tspdb.src.prediction_models.ts_svd_model import SVDModel
from math import ceil
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
import os
from datetime import datetime
from tspdb.src.pindex.pindex_utils import index_ts_mapper, index_ts_inv_mapper, index_exists, get_bound_time
from sklearn.metrics import r2_score
import time
import pickle
from sqlalchemy.types import *
from tspdb.src.tsUtils import unnormalize 

def delete_pindex(db_interface, index_name, schema='tspdb'):
    """
    Delete Pindex index_name from database.
    ----------
    Parameters
    ----------
    db_interface: DBInterface object
        instant of an interface with the db
    
    index_name: string 
        name of the pindex to be deleted

    schema: string 
        name of the tspdb schema
    """
    # suffixes of the pindex tables
    suffix = ['u', 'v', 's', 'm', 'c', 'meta']
    index_name_ = schema + '.' + index_name
    table_name = None
    try:
        meta_table = index_name_ + "_meta"
        # get time series table name
        table_name = db_interface.query_table(meta_table, columns_queried=['time_series_table_name'])[0][0]
    except: 
        pass
    index_name_no_schema = index_name.split('.')[-1]
    if table_name is not None:
        db_interface.drop_trigger(table_name,index_name_no_schema)
    # drop mean and variance tables 
    for suf in suffix:
        db_interface.drop_table(index_name_ + '_' + suf)
        db_interface.drop_table(index_name_ + '_variance_' + suf)

    # drop pindex data from pindices and oindices_coumns tables and the insert trigger on table_name
    db_interface.delete('tspdb.pindices', "index_name = '" + str(index_name) + "';")
    db_interface.delete('tspdb.pindices_columns', "index_name = '" + str(index_name) + "';")
    db_interface.delete('tspdb.pindices_stats', "index_name = '" + str(index_name) + "';")


def load_pindex_u(db_interface,index_name):
    t = time.time()
    meta_table = index_name + "_meta"
    meta_inf = db_interface.query_table(meta_table,
                                        columns_queried=['T', 'T0', 'k', 'gamma', 'var_direct_method', 'k_var', 'T_var',
                                                         'soft_thresholding', 'start_time', 'aggregation_method',
                                                         'agg_interval', 'persist_l','col_to_row_ratio', 'L','last_TS_fullSVD','last_TS_inc',
                                                              'last_TS_seen', 'p' ,'time_series_table_name', 'indexed_column','time_column'])
    
    T, T0, k, gamma, direct_var, k_var, T_var, SSVT, start_time, aggregation_method, agg_interval, persist_l, col_to_row_ratio, L, ReconIndex, MUpdateIndex, TimeSeriesIndex , p= meta_inf[0][:-3]
    L_m = db_interface.query_table(index_name + "_m", ['L'], 'modelno =0')[0][0]
    
    time_series_table_name, value_column, time_column = meta_inf[0][-3:]
    last = get_bound_time(db_interface, time_series_table_name, time_column ,'max')
    value_columns = value_column.split(',')
    # ------------------------------------------------------
    # temp fix
    gamma = float(gamma)
    if not isinstance(start_time, (int, np.integer)):
        start_time = pd.to_datetime(start_time)
    if not isinstance(last, (int, np.integer)):
        last = pd.to_datetime(start_time)
    agg_interval = float(agg_interval)
    # ------------------------------------------------------
    no_ts = len(value_columns)
    last_index = (index_ts_mapper(start_time, agg_interval, last) + 1)
    if last_index - MUpdateIndex//no_ts <= 5*L_m:
        print(L, last_index, MUpdateIndex)
        print('nothing major to update')
        return False
    if p < 1.0:
        fill_in_missing = False
    else: fill_in_missing = True
    TSPD = TSPI(interface=db_interface, index_name=index_name, schema=None, T=T, T0=T0, rank=k, gamma=gamma,
                direct_var=direct_var, rank_var=k_var, T_var=T_var, SSVT=SSVT, start_time=start_time,
                aggregation_method=aggregation_method, agg_interval=agg_interval, time_series_table_name=time_series_table_name, 
                time_column = time_column, value_column = value_columns ,persist_L = persist_l,col_to_row_ratio = col_to_row_ratio, fill_in_missing = fill_in_missing, p =p)
    
    model_no = int(max((last_index*no_ts - 1) / (T / 2) - 1, 0))
    last_model_no = int(max((MUpdateIndex - 1) / (T / 2) - 1, 0))
    model_start = last_model_no*T/2
    print(model_no, last_model_no, ReconIndex, model_start, last_index)
    
    new_points_ratio = (last_index*no_ts - ReconIndex)/(ReconIndex - model_start)
    print(new_points_ratio)
    
    if new_points_ratio < gamma and model_no <= last_model_no and (last_index*no_ts)%(T//2) != 0:
        print('marginal update')
        start = (MUpdateIndex)//TSPD.no_ts
        end = (TimeSeriesIndex - 1)//TSPD.no_ts
    else:
        print('big update')
        start = max((TimeSeriesIndex - T)//TSPD.no_ts,0)
        end = (TimeSeriesIndex - 1)//TSPD.no_ts
    # initiate TSPI object 
    TSPD.ts_model = TSMM(TSPD.k, TSPD.T, TSPD.gamma, TSPD.T0, col_to_row_ratio=col_to_row_ratio,
                         model_table_name=index_name, SSVT=TSPD.SSVT, L=L, persist_L = TSPD.persist_L, no_ts = TSPD.no_ts, fill_in_missing = fill_in_missing, p =p)
    TSPD.ts_model.ReconIndex, TSPD.ts_model.MUpdateIndex, TSPD.ts_model.TimeSeriesIndex = ReconIndex, MUpdateIndex, TimeSeriesIndex

    # load variance models if any
    if TSPD.k_var != 0:
        col_to_row_ratio, L, ReconIndex, MUpdateIndex, TimeSeriesIndex = db_interface.query_table(meta_table,
                                                                                                  columns_queried=[
                                                                                                      'col_to_row_ratio_var',
                                                                                                      'L_var',
                                                                                                      'last_TS_fullSVD_var',
                                                                                                      'last_TS_inc_var',
                                                                                                      'last_TS_seen_var'])[0]

        TSPD.var_model = TSMM(TSPD.k_var, TSPD.T_var, TSPD.gamma, TSPD.T0, col_to_row_ratio=col_to_row_ratio,
                              model_table_name=index_name + "_variance", SSVT=TSPD.SSVT, L=L, persist_L =TSPD.persist_L, no_ts = TSPD.no_ts, fill_in_missing = fill_in_missing, p =p)
        TSPD.var_model.ReconIndex, TSPD.var_model.MUpdateIndex, TSPD.var_model.TimeSeriesIndex = ReconIndex, MUpdateIndex, TimeSeriesIndex

    print('loading meta_model time', time.time()-t)
    # LOADING SUB-MODELs Information
    TSPD._load_models_from_db(TSPD.ts_model)
    print('loading sub models time', time.time()-t)
    if end >= start:
        start_point = index_ts_inv_mapper(TSPD.start_time, TSPD.agg_interval, start)
        end_point = index_ts_inv_mapper(TSPD.start_time, TSPD.agg_interval, end)
        TSPD.ts_model.TimeSeries = TSPD._get_range(start_point, end_point)
        print('loading time series time', time.time()-t)
        print(start, end, start_point,end_point)
    # query variance models table
    if TSPD.k_var != 0:
        TSPD._load_models_from_db(TSPD.var_model)

        # load last T points of  variance time series (squared of observations if not direct_var)
        if TSPD.direct_var:
            end_var = (TSPD.var_model.TimeSeriesIndex - 1)//TSPD.no_ts
            start = max(start -1,0)
            TT = min(end_var-start+1, TSPD.var_model.T//TSPD.no_ts)
            if (end_var-start+1) - TT >0:
                start +=  (end_var-start+1) - TT 
            mean = np.zeros([TT,TSPD.no_ts])
            print(mean.shape, start, end_var, TSPD.var_model.T )
            start_point = index_ts_inv_mapper(TSPD.start_time, TSPD.agg_interval, start)
            end_point = index_ts_inv_mapper(TSPD.start_time, TSPD.agg_interval, end_var)
            print(start, end_var, start_point,end_point,TT)
            if end_var != start:
                for ts_n, value_column in enumerate(TSPD.value_column):
                    mean[:,ts_n] = get_prediction_range(index_name, TSPD.time_series_table_name, value_column,db_interface, start_point, end_point, uq=False)
                TSPD.var_model.TimeSeries = TSPD.ts_model.TimeSeries[:len(mean),:] - mean
        else:
            TSPD.var_model.TimeSeries = (TSPD.ts_model.TimeSeries) ** 2
    print('loading time series variance time', time.time()-t)
    return TSPD


class TSPI(object):
    # k:                        (int) the number of singular values to retain in the means prediction model
    # k_var:                    (int) the number of singular values to retain in the variance prediction model
    # T0:                       (int) the number of entries below which the model will not be trained
    # T:                        (int) Number of entries in each submodel in the means prediction model
    # T_var:                    (int) Number of entries in each submodel in the variance prediction model
    # gamma:                    (float) (0,1) fraction of T after which the last sub-model is fully updated
    # col_to_row_ratio:         (int) the ratio of no. columns to the number of rows in each sub-model
    # L:                        (int) the number of rows in each sub-model. if set, rectFactor is ignored.
    # DBinterface:              (DBInterface object) the object used in communicating with the database.
    # table_name:               (str) The name of the time series table in the database
    # time_column:              (str) The name of the time column in the time series table in the database
    # value_column:             (str) The name of the value column in the time series table in the database
    # var_method_diff:          (bol) if True, calculate variance by subtracting the mean from the observations in the variance prediction model
    # mean_model:               (TSMM object) the means prediction model object
    # var_model:                (TSMM object) the variance prediction model object

    def __init__(self, rank = None, rank_var = 1, T=int(1e5), T_var=None, gamma=0.2, T0=100, col_to_row_ratio=10,
                 interface=Interface, agg_interval=None, start_time=None, aggregation_method='average',
                 time_series_table_name= "", time_column = "", value_column = [''], SSVT=False, p=None, direct_var=True, L=None,  recreate=True,
                 index_name=None, _dir='', schema='tspdb', persist_L = None, normalize = True, auto_update = True, fill_in_missing = True):
        if gamma <0 or gamma >=1:
            gamma = 0.5
        self._dir = _dir
        self.index_ts = False
        self.db_interface = interface
        self.time_series_table_name = time_series_table_name
        self.time_column = time_column
        self.value_column = value_column
        self.no_ts = len(value_column)
        self.k = rank
        self.SSVT = SSVT
        self.auto_update = auto_update
        ############ Temp ############
        # In current implemntation, we will assume that T_var = T
        T_var = T
        ##############################
        if T_var is None:
            T_var = T
        else:
            T_var = T_var

        self.persist_L = persist_L
        if self.persist_L is None:       
            self.persist_L = False
            if L is not None:
                self.persist_L = True
        
        self.gamma = gamma
        self.T0 = T0
        self.k_var = rank_var
        self.index_name = index_name
        self.schema = schema
        # if agg_interval is not given, estimate it from the first 100 points
        if agg_interval is None:
            agg_interval = interface.get_time_diff( self.time_series_table_name, self.time_column,100)
        
        self.agg_interval = agg_interval
        
        self.aggregation_method = aggregation_method
        self.start_time = start_time
        self.tz = None
        self.normalize = normalize
        if self.start_time is None:
            self.start_time = get_bound_time(interface, self.time_series_table_name, self.time_column, 'min')
        
        if self.index_name is None:
            self.index_name = self.schema + '.pindex_' + self.time_series_table_name
        
        elif schema is not None:
            self.index_name = self.schema + '.' + self.index_name

        if isinstance(self.start_time, (int, np.integer)):
            self.agg_interval = 1.
        self.fill_in_missing = fill_in_missing
        self.ts_model = TSMM(self.k, T, self.gamma, self.T0, col_to_row_ratio=col_to_row_ratio,
                             model_table_name=self.index_name, SSVT=self.SSVT, p=None, L=L, persist_L = self.persist_L, 
                             no_ts = self.no_ts, normalize = self.normalize, fill_in_missing = self.fill_in_missing)
        self.var_model = TSMM(self.k_var, T_var, self.gamma, self.T0, col_to_row_ratio=col_to_row_ratio,
                              model_table_name=self.index_name + "_variance", SSVT=self.SSVT, p=None,
                              L=L, persist_L = self.persist_L, no_ts = self.no_ts, normalize = self.normalize, fill_in_missing = self.fill_in_missing)
        self.direct_var = direct_var
        self.T = self.ts_model.T
        self.T_var = self.var_model.T
        if self.k_var:
            self.uq = True
        else:
            self.uq = False
        self.norm_mean = np.zeros(self.no_ts)
        self.norm_std = np.ones(self.no_ts)

    def create_index(self):
        """
        This function query new datapoints from the database using the variable self.TimeSeriesIndex and call the
        update_model function
        """
        # find starting and ending time 
        end_point = get_bound_time(self.db_interface, self.time_series_table_name, self.time_column, 'max')
        start_point = index_ts_inv_mapper(self.start_time, self.agg_interval, self.ts_model.TimeSeriesIndex)
        
        # get new entries
        new_entries = self._get_range(start_point, end_point)
        new_entries = new_entries.astype('float')
        if len(new_entries) > 0:
            self.update_model(new_entries)
            self.write_model(True)
        
        # drop and create trigger
        if self.auto_update:
            self.db_interface.create_insert_trigger(self.time_series_table_name, self.index_name)

    def update_index(self):
        """
        This function query new datapoints from the database using the variable self.TimeSeriesIndex and call the
        update_model function
        """
        end_point = get_bound_time(self.db_interface, self.time_series_table_name, self.time_column, 'max')
        start_point = index_ts_inv_mapper(self.start_time, self.agg_interval, self.ts_model.TimeSeriesIndex//self.no_ts)
        new_entries =  np.array(self._get_range(start_point, end_point), dtype = np.float)
        if len(new_entries) > 0:
            self.update_model(new_entries)
            self.write_model(False)

    def update_model(self, NewEntries):
        """
        This function takes a new set of entries and update the model accordingly.
        if the number of new entries means new model need to be bulit, this function segment the new entries into
        several entries and then feed them to the update_ts and fit function
        :param NewEntries: Entries to be included in the new model
        """
        # ------------------------------------------------------
        # is it already numpy.array? ( not really needed but not harmful)
        obs = np.array(NewEntries).astype('float')
        if NewEntries.size == 0:
            return
        # ------------------------------------------------------
        # lag is the the slack between the variance and timeseries model        
        lag = None
        if self.ts_model.TimeSeries is not None:
            lag = (self.ts_model.TimeSeriesIndex//self.no_ts - self.var_model.TimeSeriesIndex//self.no_ts)
            if lag > 0:
                lagged_obs = self.ts_model.TimeSeries[-lag:,:]
            else: lag = None

        # Update mean model
        self.ts_model.update_model(NewEntries)
        
        self.k = self.ts_model.kSingularValuesToKeep
        # Determine updated models
        
        models = {k: self.ts_model.models[k] for k in self.ts_model.models if self.ts_model.models[k].updated}

        if self.k_var:
            if self.direct_var:

                means = self.ts_model._denoiseTS(models)[self.var_model.TimeSeriesIndex//self.no_ts:self.ts_model.MUpdateIndex//self.no_ts,:]
                if lag is not None:
                    var_obs = np.concatenate([lagged_obs, obs])
                else:
                    var_obs = obs
                print(obs.shape, self.ts_model.MUpdateIndex, self.var_model.TimeSeriesIndex, self.ts_model._denoiseTS(models).shape,means.shape,var_obs.shape)
                var_entries = np.square(var_obs[:len(means),:] - means)
                # ------------------------------------------------------
                # EDIT: Is this necessary (NAN to zero)?
                # var_entries[np.isnan(var_obs[:len(means)])] = 0    
                # ------------------------------------------------------
                self.var_model.update_model(var_entries)
            else:
                var_entries = np.square(NewEntries)
                self.var_model.update_model(var_entries)

    def write_model(self, create):
        """
        write the pindex to db
        ----------
        Parameters
        ----------
        create: bol 
            if Ture, create the index in DB, else update it.
        """

        # remove schema name if exist
        t = time.time()
        index_name = self.index_name.split('.')[1]

        # delete meta data if create
        if create:
            delete_pindex(self.db_interface, index_name)
    
        # write mean and variance tables
        self.write_tsmm_model(self.ts_model, create)
        self.write_tsmm_model(self.var_model, create)
        self.calculate_out_of_sample_error(self.ts_model)
        # if time is timestamp, convert to pd.Timestamp
        if not isinstance(self.start_time, (int, np.integer)):
            self.start_time = pd.to_datetime(self.start_time)

        # prepare meta data table
        metadf = pd.DataFrame(
            data={'T': [self.ts_model.T], 'T0': [self.T0], 'gamma': [float(self.gamma)], 'k': [self.k],
                  'L': [self.ts_model.L],
                  'last_TS_seen': [self.ts_model.TimeSeriesIndex], 'last_TS_inc': [self.ts_model.MUpdateIndex],
                  'last_TS_fullSVD': [self.ts_model.ReconIndex],
                  'time_series_table_name': [self.time_series_table_name], 'indexed_column': [','.join(self.value_column)],
                  'time_column': [self.time_column],
                  'soft_thresholding': [self.SSVT], 'no_submodels': [len(self.ts_model.models)],
                  'no_submodels_var': [len(self.var_model.models)],
                  'col_to_row_ratio': [self.ts_model.col_to_row_ratio],
                  'col_to_row_ratio_var': [self.var_model.col_to_row_ratio], 'T_var': [self.var_model.T],
                  'k_var': [self.k_var], 'L_var': [self.var_model.L],
                  'last_TS_seen_var': [self.var_model.TimeSeriesIndex],
                  'last_TS_inc_var': [self.var_model.MUpdateIndex], 'aggregation_method': [self.aggregation_method],
                  'agg_interval': [self.agg_interval],
                  'start_time': [self.start_time], 'last_TS_fullSVD_var': [self.var_model.ReconIndex],
                  'var_direct_method': [self.direct_var], 'persist_l': [self.persist_L], 'p': [self.ts_model.p]})
        
        # ------------------------------------------------------
        # EDIT: Due to some incompatibiliy with PSQL timestamp types 
        # Further investigate 
        # ------------------------------------------------------
        if not isinstance(self.start_time, (int, np.integer)):
            #metadf['start_time'] = metadf['start_time'].astype(pd.Timestamp)
            metadf['start_time'] = metadf['start_time'].astype('datetime64[ns]')
        last_index = index_ts_inv_mapper(self.start_time, self.agg_interval, self.ts_model.TimeSeriesIndex//self.no_ts -1)
        if create:
            # create meta table
            self.db_interface.create_table(self.index_name + '_meta', metadf, include_index=False)
            
            # populate column pindices
            for i,ts in enumerate(self.value_column):
                self.db_interface.insert('tspdb.pindices_columns', [index_name, ts],
                                     columns=['index_name', 'value_column'])

        else:
            # else update meta table, tspdb pindices 
            self.db_interface.delete(self.index_name + '_meta', '')
            self.db_interface.insert(self.index_name + '_meta', metadf.iloc[0])
            self.db_interface.delete('tspdb.pindices', "index_name = '" + str(index_name) + "';")
            self.db_interface.delete('tspdb.pindices_stats', "index_name = '" + str(index_name) + "';")
            
            # UPDATE STAT TABLE
        for i,ts in enumerate(self.value_column):
            forecast_tests_array = np.array([m.forecast_model_score_test[i] for m in self.ts_model.models.values()],'float')
            self.db_interface.insert('tspdb.pindices_stats',
                                     [index_name, ts, self.ts_model.TimeSeriesIndex//self.no_ts, len(self.ts_model.models),np.mean([ m.imputation_model_score[i] for m in self.ts_model.models.values() ]), np.mean([ m.forecast_model_score[i] for m in self.ts_model.models.values()]),np.nanmean(forecast_tests_array)],
                                     columns=['index_name', 'column_name','number_of_observations', 'number_of_trained_models', 'imputation_score', 'forecast_score','test_forecast_score'])
            
            # UPDATE PINDICES TABLE
        if isinstance(self.start_time, (int, np.integer)):
            self.db_interface.insert('tspdb.pindices',
                                     [index_name, self.time_series_table_name, self.time_column, self.uq,
                                      self.agg_interval, self.start_time, last_index],
                                     columns=['index_name', 'relation', 'time_column', 'uq', 'agg_interval',
                                              'initial_index', 'last_index'])
        else:
            self.db_interface.insert('tspdb.pindices',
                                     [index_name, self.time_series_table_name, self.time_column, self.uq,
                                      self.agg_interval, self.start_time, last_index],
                                     columns=['index_name', 'relation', 'time_column', 'uq', 'agg_interval',
                                              'initial_timestamp', 'last_timestamp'])
    
    def prepare_tsmm_to_store(self):
        for tsmm in [self.ts_model, self.var_model]:
            no_models = len(tsmm.models)
            # remove matrix from the last sub-model
            tsmm.models[no_models-1].matrix = None
            if no_models<2: return
            # remove matrix, Uk, Vk, s, Ukw, Vkw from all sub-models except the last one
            for m in range(no_models-1):
                tsmm.models[m].matrix = None
                tsmm.models[m].Uk = None
                tsmm.models[m].Vk = None
                tsmm.models[m].sk = None
                tsmm.models[m].Ukw = None
                tsmm.models[m].Vkw = None
                tsmm.models[m].skw = None
                tsmm.models[m].weights = None
                tsmm.models[m].updated = False 

    def write_tsmm_model(self, tsmm, create):
        """
        -
        """
        ########################### To Do ######################
        # 1 Replace for loops with vectorized numpy operations?
        ########################################################

        # only get updated sub models
        models = {k: tsmm.models[k] for k in tsmm.models if tsmm.models[k].updated}

        # Mo
        model_name = tsmm.model_tables_name

        if len(models) == 0:
            return

        
        tableNames = [model_name + '_' + c for c in ['u', 'v', 's', 'c', 'm']]

        last_model = max(models.keys())
        first_model = min(models.keys())
        N = models[first_model].N
        M = models[first_model].M
        
        if last_model == first_model:
            N = tsmm.models[0].N
            M = tsmm.models[0].M
            
        # populate U_table data
        U_table = np.zeros([(len(models) - 1) * N + models[last_model].N, 1 + 2*tsmm.kSingularValuesToKeep])
        for i, m in sorted(models.items()):
            j = i - first_model
            if i == last_model:
                U_table[j * N:, 1:1 + tsmm.kSingularValuesToKeep] = m.Uk
                U_table[j * N:, 1 + tsmm.kSingularValuesToKeep: 1 + 2*tsmm.kSingularValuesToKeep] = np.concatenate((m.Ukw,np.zeros([1,tsmm.kSingularValuesToKeep])))
                U_table[j * N:, 0] = int(i)
            else:
                U_table[j * N:(j + 1) * N, 1:1 + tsmm.kSingularValuesToKeep] = m.Uk
                U_table[j * N:(j + 1) * N, 1 + tsmm.kSingularValuesToKeep:1 + 2*tsmm.kSingularValuesToKeep] =  np.concatenate((m.Ukw,np.zeros([1,tsmm.kSingularValuesToKeep])))
                U_table[j * N:(j + 1) * N, 0] = int(i)

        columns = ['modelno'] + ['u' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)]+ ['uw' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)]
        udf = pd.DataFrame(columns=columns, data=U_table)
        udf.index = np.arange(first_model * N, first_model * N + len(U_table))
        udf['tsrow'] = (udf.index % N).astype(int)

        if create:
            self.db_interface.create_table(tableNames[0], udf, 'row_id', index_label='row_id')
        else:
            self.db_interface.delete(tableNames[0], 'modelno >= %s and modelno <= %s' % (first_model, last_model,))
            self.db_interface.bulk_insert(tableNames[0], udf, index_label='row_id')

        # populate V_table data
        V_table = np.zeros([(len(models) - 1) * M + models[last_model].M, 1 + 2*tsmm.kSingularValuesToKeep])
        for i, m in sorted(models.items()):
            j = i - first_model
            if i == last_model:
                V_table[j * M:, 1:1 + tsmm.kSingularValuesToKeep] = m.Vk
                V_table[j * M:, 1 + tsmm.kSingularValuesToKeep: 1+ 2*tsmm.kSingularValuesToKeep] = m.Vkw
                V_table[j * M:, 0] = int(i)

            else:
                V_table[j * M:(j + 1) * M, 1:1 + tsmm.kSingularValuesToKeep] = m.Vk
                V_table[j * M:(j + 1) * M, 1 + tsmm.kSingularValuesToKeep: 1+ 2*tsmm.kSingularValuesToKeep] = m.Vkw
                V_table[j * M:(j + 1) * M, 0] = int(i)

        columns = ['modelno'] + ['v' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)] + ['vw' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)]
        vdf = pd.DataFrame(columns=columns, data=V_table)
        vdf.index = np.arange(first_model * M, first_model * M + len(V_table))
        vdf['tscolumn'] = (vdf.index - 0.5 * M * vdf['modelno']).astype(int)
        vdf['time_series'] = (vdf.index%self.no_ts).astype(int)
        
        if create:
            self.db_interface.create_table(tableNames[1], vdf, 'row_id', index_label='row_id')
        else:
            self.db_interface.delete(tableNames[1], 'modelno >= %s and modelno <= %s' % (first_model, last_model,))
            self.db_interface.bulk_insert(tableNames[1], vdf, index_label='row_id')

        # populate s_table data 
        s_table = np.zeros([len(models), 1 + 2*tsmm.kSingularValuesToKeep])
        for i, m in sorted(models.items()):
            j = i - first_model
            s_table[j, 1:tsmm.kSingularValuesToKeep + 1] = m.sk
            s_table[j, tsmm.kSingularValuesToKeep + 1:tsmm.kSingularValuesToKeep*2 + 1] = m.skw
            s_table[j, 0] = int(i)
        columns = ['modelno'] + ['s' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)] + ['sw' + str(i) for i in range(1, tsmm.kSingularValuesToKeep + 1)]
        sdf = pd.DataFrame(columns=columns, data=s_table)
        if create:
            self.db_interface.create_table(tableNames[2], sdf, 'modelno', include_index=False, index_label='row_id')
        else:
            self.db_interface.delete(tableNames[2], 'modelno >= %s and modelno <= %s' % (first_model, last_model,))
            self.db_interface.bulk_insert(tableNames[2], sdf, include_index=False)

        # populate c_table data 
        id_c = 0
        w_f = N - 1
        w_l = len(models[last_model].weights)
        c_table = np.zeros([(len(models)) * (w_f+self.no_ts), 3])
        for i,m in sorted(models.items()):
            for ts in range(self.no_ts):
                bias = (-m.weights.sum() +1)*m.norm_mean[ts]
                c_table[id_c, :] = [i, -ts-1, bias]
                id_c += 1
            coeNu = 0
            for weig in m.weights[::-1]:
                c_table[id_c, :] = [i, coeNu, weig]
                id_c += 1
                coeNu += 1
            if i == last_model:
                for q in range(w_f-w_l):
                    c_table[id_c, :] = [i, coeNu, 0]
                    id_c += 1
                    coeNu += 1


        cdf = pd.DataFrame(columns=['modelno', 'coeffpos', 'coeffvalue'], data=c_table)
        cdf.index = np.arange(first_model * (w_f+self.no_ts), first_model * (w_f+self.no_ts) + len(c_table))

        if create:
            self.db_interface.create_table(tableNames[3], cdf, 'row_id', index_label='row_id')
        else:
            self.db_interface.delete(tableNames[3], 'modelno >= %s and modelno <= %s' % (first_model, last_model,))
            self.db_interface.bulk_insert(tableNames[3], cdf, include_index=True, index_label="row_id")

        # populate m_table data 
        m_table = np.zeros([len(models), 7 + 5], 'object')
        for i, m in sorted(models.items()):
            m_table[i - first_model, :] = [i, m.N, m.M, m.start, m.M * m.N, m.TimesUpdated, m.TimesReconstructed, self._array_str(list(m.imputation_model_score)), self._array_str(list(m.forecast_model_score)), self._array_str(list(m.forecast_model_score_test)), self._array_str(list(m.norm_mean)), self._array_str(list(m.norm_std))]
            #for ts in range(self.no_ts):
            #        m_table[i - first_model,7+5*ts:7+5*(ts+1)] = [ m.imputation_model_score[ts], m.forecast_model_score[ts], np.nan, m.norm_mean[ts], m.norm_std[ts]]
        model_table_col = ['modelno', 'L', 'N', 'start', 'dataPoints', 'timesUpdated', 'timesRecons']
        model_table_col += ['imputation_acc', 'forecasting_acc', 'forecasting_test_acc','norm_mean', 'norm_std']
        
        # model_table_acc_label = ['imputation_acc_%s', 'forecasting_acc_%s', 'forecasting_test_acc_%s']
        # model_table_norm_label = ['norm_mean_%s', 'norm_std_%s']
        # for ts in range(self.no_ts):
        #     model_table_col += [i%(self.value_column[ts]) for i in model_table_acc_label]
        #     model_table_col += [i%(self.value_column[ts]) for i in model_table_norm_label]
        types = [Integer(),Integer(),Integer(),Integer(),Integer(),Integer(),Integer(),ARRAY(Float),ARRAY(Float),ARRAY(Float),ARRAY(Float),ARRAY(Float)]
        mdf = pd.DataFrame(columns=model_table_col,
                           data=list(m_table))
        type_dict = {model_table_col[i]: types[i] for i in range(len(model_table_col))}
        if create:
            self.db_interface.create_table(tableNames[4], mdf, 'modelno', include_index=False, index_label='modelno', type_dict = type_dict)
        else:
            self.db_interface.delete(tableNames[4], 'modelno >= %s and modelno <= %s' % (first_model, last_model,))
            self.db_interface.bulk_insert(tableNames[4], mdf, include_index=False)

        if create:
            self.db_interface.create_index(tableNames[0], 'tsrow, modelno')
            self.db_interface.create_index(tableNames[0], 'modelno')
            self.db_interface.create_index(tableNames[1], 'tscolumn, modelno')
            self.db_interface.create_index(tableNames[1], 'modelno')
            self.db_interface.create_index(tableNames[2], 'modelno')
            self.db_interface.create_index(tableNames[3], 'modelno')
            self.db_interface.create_index(tableNames[3], 'coeffpos')
            self.db_interface.create_coefficients_average_table(tableNames[3], tableNames[3] + '_view', [1,2,10, 20, 100],
                                                                last_model)
            self.db_interface.create_index(tableNames[3] + '_view', 'coeffpos')
        else:
            last_model = len(tsmm.models) - 1
            self.db_interface.create_coefficients_average_table(tableNames[3], tableNames[3] + '_view', [1,2,10, 20, 100],
                                                                last_model, refresh=True)
        
    
    def calculate_out_of_sample_error(self, tsmm):
        models = {k: tsmm.models[k] for k in tsmm.models if tsmm.models[k].updated}
        if len(models.keys()) == 0:
            return 
        last_model = max(models.keys())
        index_name = tsmm.model_tables_name
        m_table = np.zeros([len(models),7+5],object)
        columns = ['modelno', 'L', 'N', 'start', 'dataPoints', 'timesUpdated', 'timesRecons']
        columns+= ['imputation_acc', 'forecasting_acc', 'forecasting_test_acc','norm_mean', 'norm_std']
        
        # model_table_acc_label = ['imputation_acc_%s', 'forecasting_acc_%s', 'forecasting_test_acc_%s']
        # model_table_norm_label = ['norm_mean_%s', 'norm_std_%s']
        # for ts in range(self.no_ts):
        #     columns += [i%(self.value_column[ts]) for i in model_table_acc_label]
        #     columns += [i%(self.value_column[ts]) for i in model_table_norm_label]
        i = 0
        for model in models:
            if model <=1 or model == last_model: continue
            matrix = np.array(models[model].matrix[:-1,:])
            L = matrix.shape[0]
            coeffs = self.db_interface.get_coeff_model(index_name+'_c',model-2 )
            coeffs_ts = coeffs[-self.no_ts:]
            coeffs = coeffs[:-self.no_ts]
            model_row = np.array(self.db_interface.query_table(index_name+'_m',columns,'modelno = %s'%(model-2))[0],'object')
            index = 9
            for ts in range(self.no_ts):
                matrix_ = unnormalize(np.array(matrix[:,ts::self.no_ts].T), models[model].norm_mean[ts], models[model].norm_std[ts])
                y_h = np.dot(matrix_, coeffs[:])
                y_h = y_h + coeffs_ts[ts] 
                y = unnormalize(models[model].lastRowObservations[ts::self.no_ts], models[model].norm_mean[ts], models[model].norm_std[ts])
                out_of_sample_error = r2_score(y,y_h)
                tsmm.models[model-2].forecast_model_score_test[ts] = out_of_sample_error
            self.db_interface.delete(index_name+'_m', 'modelno = %s' % (model-2,))
            model_row[index] = self._array_str(list(tsmm.models[model-2].forecast_model_score_test))
            for ii in [7,8,10,11]:
                model_row[ii] = self._array_str(list(model_row[ii]))
            

            m_table[i,:] = model_row
            i+=1
        m_table = m_table[:i,:]
       
        mdf = pd.DataFrame(columns=columns,
                           data=list(m_table))
       
        self.db_interface.bulk_insert(index_name+'_m', mdf, include_index=False)

    def _array_str(self,array_):
        string = str(array_)
        return '{'+string[1:-1]+'}'

    def _get_range(self, t1, t2=None):
        """
        implement the same singles point query. use get from table function in interface
        """

        return pd.DataFrame(self.db_interface.get_time_series(self.time_series_table_name, t1, t2,
                                                                         value_column=','.join(self.value_column),
                                                                         index_column=self.time_column,
                                                                         aggregation_method=self.aggregation_method,
                                                                         interval=self.agg_interval,
                                                                         start_ts=self.start_time)).values


    def _load_models_from_db(self, tsmm):

        models_info_table = tsmm.model_tables_name + '_m'
        columns = ['modelno','L','N','start','timesUpdated','timesRecons']
        columns+= ['imputation_acc', 'forecasting_acc', 'forecasting_test_acc','norm_mean', 'norm_std']
        # model_table_ts_label = ['imputation_acc_%s', 'forecasting_acc_%s', 'forecasting_test_acc_%s','norm_mean_%s', 'norm_std_%s']
        # for ts in range(self.no_ts):
        #     columns += [i%(self.value_column[ts]) for i in model_table_ts_label]
        
        info = self.db_interface.query_table(models_info_table, columns_queried=columns)
        # info = self.db_interface.query_table(models_info_table, columns_queried=['*'])
        for model in info:
            tsmm.models[int(model[0])] = SVDModel('t1', tsmm.kSingularValuesToKeep, int(model[1]), int(model[2]),
                                                  start=int(model[3]),
                                                  TimesReconstructed=int(model[5]),
                                                  TimesUpdated=int(model[4]), SSVT=tsmm.SSVT, probObservation=tsmm.p,
                                                  updated=False, no_ts = self.no_ts, imputation_model_score = list(model[6]),  forecast_model_score = list(model[7]), forecast_model_score_test = list(model[8]),\
                                                  norm_mean = list(model[9]) , norm_std = list(model[10]))
        # load last model
        last_model = len(tsmm.models) - 1
        S= self.db_interface.get_S_row(tsmm.model_tables_name + '_s', [last_model, last_model],tsmm.kSingularValuesToKeep, return_weights_decom = True)[0]
        U = self.db_interface.get_U_row(tsmm.model_tables_name + '_u', [0, 2 * tsmm.L],[last_model, last_model], tsmm.kSingularValuesToKeep,return_weights_decom = True)
        V = self.db_interface.get_V_row(tsmm.model_tables_name + '_v',[0, tsmm.TimeSeriesIndex], tsmm.kSingularValuesToKeep, None, models_range = [last_model, last_model], return_weights_decom = True)
        print(V.shape,U.shape)

        tsmm.models[last_model].sk = S[:tsmm.kSingularValuesToKeep]
        tsmm.models[last_model].Uk = U[:,:tsmm.kSingularValuesToKeep]
        tsmm.models[last_model].Vk = V[:,:tsmm.kSingularValuesToKeep]
        tsmm.models[last_model].skw = S[tsmm.kSingularValuesToKeep:]
        tsmm.models[last_model].Ukw = U[:-1,tsmm.kSingularValuesToKeep:]
        tsmm.models[last_model].Vkw = V[:,tsmm.kSingularValuesToKeep:]

 