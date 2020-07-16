import numpy as np
import pandas as pd
from tspdb.src.database_module.db_class import Interface
from tspdb.src.pindex.pindex_utils import index_ts_mapper, index_ts_inv_mapper, index_exists, get_bound_time
from scipy.stats import norm

def unnormalize(arr, mean, std):
    return arr *std + mean

def get_prediction_range( index_name, table_name, value_column, interface, t1,t2 , uq = True, uq_method ='Gaussian', c = 95., projected = False):

    """
    Return an array of N (N = t2-t1+1) predicted value along with the confidence interval for the value of column_name  at time t1 to t2  
    using index_name  by calling either forecast_range or impute_range function 
    ----------
    Parameters
    ----------
    index_name: string 
        name of the PINDEX used to query the prediction

    table_name: string 
        name of the time series table in the database

    value_column: string
        name of column than contain time series value

    interface: db_class object
        object used to communicate with the DB. see ../database/db_class for the abstract class
    
    t1: (int or timestamp)
        index or timestamp indicating the start of the queried range 
    
    t2: (int or timestamp)
        index or timestamp indicating the end of the queried range 
    
    uq: boolean optional (default=true) 
        if true,  return upper and lower bound of the  c% confidenc interval

    uq_method: string optional (defalut = 'Gaussian') options: {'Gaussian', 'Chebyshev'}
        Uncertainty quantification method used to estimate the confidence interval

    c: float optional (default 95.)    
        confidence level for uncertainty quantification, 0<c<100
    ----------
    Returns
    ----------
    prediction array, shape [(t1 - t2 +1)  ]
        Values of the predicted point of the time series in the time interval t1 to t2
    
    deviation array, shape [1, (t1 - t2 +1)  ]
        The deviation from the mean to get the desired confidence level 
    """
    # query pindex parameters


    T,T_var, L, k,k_var, L_var, last_model, MUpdateIndex,var_direct, interval, start_ts, last_TS_seen, last_TS_seen_var, index_col, value_columns, MUpdateIndex_var, p = interface.query_table( index_name+'_meta',['T','T_var', 'L', 'k','k_var','L_var', 'no_submodels', 'last_TS_inc', 'var_direct_method', 'agg_interval','start_time', "last_TS_seen", "last_TS_seen_var", "time_column","indexed_column",'last_TS_inc_var','p'])[0]
    last_model -= 1
    value_columns = value_columns.split(',')
    no_ts = len(value_columns)

    try: value_index = value_columns.index(value_column)
    except: raise Exception('The value column %s selected is not indexed by the chosen pindex'%(value_column))
    
    if not isinstance(t1, (int, np.integer)):
        t1 = pd.to_datetime(t1)
        t2 = pd.to_datetime(t2)
        start_ts = pd.to_datetime(start_ts)
    
    
    interval = float(interval)
    t1 = index_ts_mapper(start_ts, interval, t1)
    t2 = index_ts_mapper(start_ts, interval, t2)
    
    # if the models is not fit, return the mean
    if MUpdateIndex == 0:
        last_TS_seen = get_bound_time(interface, table_name, index_col, 'max')
        obs = interface.get_time_series(table_name, start_ts, last_TS_seen, start_ts = start_ts,  value_column=value_column, index_column= index_col, Desc=False, interval = interval, aggregation_method = 'average')
        if uq: return np.mean(obs)*np.ones(t2-t1+1), np.zeros(t2-t1+1)
        else: return np.mean(obs)*np.ones(t2-t1+1)


    # check uq variables
    if uq:

        if c < 0 or c >=100:
            raise Exception('confidence interval c must be in the range (0,100): 0 <=c< 100')

        if uq_method == 'Chebyshev':
            alpha = 1./(np.sqrt(1-c/100))
        elif uq_method == 'Gaussian':
            alpha = norm.ppf(1/2 + c/200)
        else:
            raise Exception('uq_method option is not recognized,  available options are: "Gaussian" or "Chebyshev"')
            
    # if all points are in the future, use _get_forecast_range 
    if t1 > (MUpdateIndex - 1)//no_ts:
        print('forecasting')
        if not uq: return _get_forecast_range(index_name,table_name, value_column, index_col, interface, t1,t2, MUpdateIndex,L,k,T,last_model,interval, start_ts, last_TS_seen,no_ts,value_index, projected = projected, p = p)
        
        else:
            prediction = _get_forecast_range(index_name,table_name, value_column, index_col, interface, t1,t2, MUpdateIndex,L,k,T,last_model,interval, start_ts, last_TS_seen,no_ts,value_index, projected = projected, p = p)
            var = _get_forecast_range(index_name+'_variance',table_name, value_column, index_col, interface, t1,t2, MUpdateIndex_var, L,k_var,T_var,last_model,interval, start_ts, last_TS_seen_var, no_ts,value_index,variance = True, direct_var =var_direct,  projected = projected,p = p)
            # if the second model is used for the second moment, subtract the squared mean to estimate the variance
            if not var_direct:
                var = var - (prediction)**2
            var *= (var>0) 
            
            return prediction, alpha*np.sqrt(var)
    
    # if all points are in the past, use get_imputation_range
    elif t2 <=  (MUpdateIndex - 1)//no_ts:    
        if not uq: return _get_imputation_range(index_name, table_name, value_column, index_col, interface, t1,t2,L,k,T,last_model, value_index, no_ts,p = p)
        else:
            prediction = _get_imputation_range(index_name, table_name, value_column, index_col, interface, t1,t2,L,k,T,last_model, value_index, no_ts,p = p)
            if (MUpdateIndex_var-1)//no_ts >= t2:
                var = _get_imputation_range(index_name+'_variance',table_name, value_column, index_col, interface, t1,t2, L_var,k_var,T_var,last_model, value_index, no_ts,p = p)
            else:
                imputations_var = _get_imputation_range(index_name+'_variance', table_name, value_column, index_col, interface, t1,(MUpdateIndex_var-1)//no_ts,L_var,k_var,T_var,last_model, value_index, no_ts,p = p)
                forecast_var = _get_forecast_range(index_name+'_variance',table_name, value_column, index_col, interface,MUpdateIndex_var//no_ts ,t2, MUpdateIndex_var,L_var,k_var,T_var,last_model,interval, start_ts,last_TS_seen, no_ts,value_index,variance = True, direct_var =var_direct,projected = projected,p = p)
                var = np.array(list(imputations_var)+list(forecast_var))
            # if the second model is used for the second moment, subtract the squared mean to estimate the variance
            if not var_direct:
                var = var - (prediction)**2
            var *= (var>0) 
            return prediction, alpha*np.sqrt(var)
    
    # if points are in both the future and in the past, use both        
    else:
        imputations = _get_imputation_range(index_name, table_name, value_column, index_col, interface, t1,(MUpdateIndex-1)//no_ts,L,k,T,last_model,value_index, no_ts,p = p)
        forecast = _get_forecast_range(index_name,table_name, value_column, index_col, interface,(MUpdateIndex)//no_ts ,t2, MUpdateIndex,L,k,T,last_model,interval, start_ts,last_TS_seen, no_ts,value_index,projected = projected,p = p)
        if not uq: return list(imputations)+list(forecast)
        else:
            imputations_var = _get_imputation_range(index_name+'_variance', table_name, value_column, index_col, interface, t1,(MUpdateIndex_var-1)//no_ts,L_var,k_var,T_var,last_model, value_index, no_ts,p = p)
            forecast_var = _get_forecast_range(index_name+'_variance',table_name, value_column, index_col, interface,MUpdateIndex_var//no_ts ,t2, MUpdateIndex_var,L_var,k_var,T_var,last_model,interval, start_ts,last_TS_seen, no_ts,value_index,variance = True, direct_var =var_direct,projected = projected,p = p)
            if not var_direct:
                forecast_var = forecast_var - (forecast)**2
                imputations_var = imputations_var - (imputations)**2
            imputations_var *= (imputations_var>0)
            forecast_var *= (forecast_var>0)
            return np.array(list(imputations)+list(forecast)), np.array(list(alpha*np.sqrt(imputations_var)) + list(alpha*np.sqrt(forecast_var)))


            


def get_prediction(index_name, table_name, value_column, interface, t, uq = True, uq_method ='Gaussian', c = 95, projected = False):
    """
    Return the predicted value along with the confidence interval for the value of column_name  at time t  using index_name 
    by calling either get_forecast or get_imputation function 
    ----------
    Parameters
    ----------
    index_name: string 
        name of the PINDEX used to query the prediction

    index_name: table_name 
        name of the time series table in the database

    value_column: string
        name of column than contain time series value

    interface: db_class object
        object used to communicate with the DB. see ../database/db_class for the abstract class
    
    t: (int or timestamp)
        index or timestamp indicating the queried time. 
    
    uq: boolean optional (default=true) 
        if true,  return upper and lower bound of the  c% confidenc interval

    uq_method: string optional (defalut = 'Gaussian') options: {'Gaussian', 'Chebyshev'}
        Uncertainty quantification method used to estimate the confidence interval

    c: float optional (default 95.)    
        confidence level for uncertainty quantification, 0<c<100
    ----------
    Returns
    ----------
    prediction float
        Values of time series at time t
    
    deviation float
        The deviation from the mean to get the desired confidence level 
    
    """
    # query pindex parameters
    
    T,T_var, L, k,k_var, L_var, last_model, MUpdateIndex,var_direct, interval, start_ts, last_TS_seen, last_TS_seen_var, index_col, value_columns, MUpdateIndex_var, p = interface.query_table( index_name+'_meta',['T','T_var', 'L', 'k','k_var','L_var', 'no_submodels', 'last_TS_inc', 'var_direct_method', 'agg_interval','start_time', "last_TS_seen", "last_TS_seen_var", "time_column","indexed_column",'last_TS_inc_var','p'])[0]
    ############ Fix queried values ####################
    last_model -= 1
    value_columns = value_columns.split(',')
    no_ts = len(value_columns)
    
    if not isinstance(t, (int, np.integer)):
        t = pd.to_datetime(t)
        start_ts = pd.to_datetime(start_ts)
    interval = float(interval)
    ###################################################
    # Check 1: value colmn is indexed 

    try: value_index = value_columns.index(value_column)
    except: raise Exception('The value column selected is not indexed by the chosen pindex')
    
    # if the model is not fit, return the average
    if MUpdateIndex == 0:
        last_TS_seen = get_bound_time(interface, table_name, index_col, 'max')
        obs = interface.get_time_series(table_name, start_ts, last_TS_seen, start_ts = start_ts,  value_column=value_column, index_column= index_col, Desc=False, interval = interval, aggregation_method = 'average')
        if uq: return np.mean(obs), 0
        else: return np.mean(obs)

    t = index_ts_mapper(start_ts, interval, t)
    if uq:
        
        if uq_method == 'Chebyshev':
            alpha = 1./(np.sqrt(1-c/100))
        
        elif uq_method == 'Gaussian':
            alpha = norm.ppf(1/2 + c/200)
        
        else:
            raise Exception('uq_method option is not recognized,  available options are: "Gaussian" or "Chebyshev"')

    if t > (MUpdateIndex - 1)//no_ts:
        if not uq: return _get_forecast_range(index_name,table_name, value_column, index_col, interface,t, t, MUpdateIndex,L,k,T,last_model, interval, start_ts, last_TS_seen,no_ts,value_index, projected = projected,p = p)[-1]
        else:
            prediction = _get_forecast_range(index_name,table_name, value_column, index_col, interface,t, t, MUpdateIndex,L,k,T,last_model, interval, start_ts,last_TS_seen, no_ts,value_index, projected = projected,p = p)[-1]
            var = _get_forecast_range(index_name+'_variance',table_name, value_column, index_col, interface,t, t, MUpdateIndex_var,L_var,k_var,T_var,last_model,interval, start_ts,last_TS_seen_var,no_ts,value_index,  projected = projected, variance = True, direct_var =var_direct,p = p)[-1]
            
            if not var_direct:
                var = var - (prediction)**2
            var *= (var>0)
            return prediction, alpha*np.sqrt(var)

    else:
        if not uq: return _get_imputation(index_name, table_name, value_column, index_col, interface, t,L,k,T,last_model,no_ts,value_index,p = p)
        else:
            prediction = _get_imputation(index_name, table_name, value_column, index_col, interface, t,L,k,T,last_model, no_ts,value_index,p = p)
            if t > (MUpdateIndex_var - 1)//no_ts: var =  _get_forecast_range(index_name+'_variance',table_name, value_column, index_col, interface,t, t, MUpdateIndex_var,L_var,k_var,T_var,last_model,interval, start_ts,last_TS_seen_var,no_ts,value_index,  projected = projected, variance = True, direct_var =var_direct, p = p)[-1]
            else: var = _get_imputation(index_name+'_variance',table_name, value_column, index_col, interface, t, L_var,k_var,T_var,last_model, no_ts,value_index,p = p)
            
            if not var_direct:
                var = var - (prediction)**2
            var *= (var>0)
            return prediction, alpha*np.sqrt(var)
            


def _get_imputation_range(index_name, table_name, value_column, index_col, interface, t1,t2,L,k,T,last_model, value_index, no_ts, p = 1.0):

    """
    Return the imputed value in the past at the time range t1 to t2 for the value of column_name using index_name 
    ----------
    Parameters
    ----------
    index_name: string 
        name of the PINDEX used to query the prediction

    index_name: table_name 
        name of the time series table in the database

    value_column: string
        name of column than contain time series value

    index_col: string  
        name of column that contains time series index/timestamp

    interface: db_class object
        object used to communicate with the DB. see ../database/db_class for the abstract class
    
    t1: (int or timestamp)
        index or timestamp indicating the start of the queried range 
    
    t2: (int or timestamp)
        index or timestamp indicating the end of the queried range  
    
    L: (int)
        Model parameter determining the number of rows in each matrix in a sub model. 
    
    k: (int )
        Model parameter determining the number of retained singular values in each matrix in a sub model. 
    
    T: (int or timestamp)
        Model parameter determining the number of datapoints in each matrix in a sub model.
    
    last_model: (int or timestamp)
        The index of the last sub model
    ----------
    Returns
    ----------
    prediction  array, shape [(t1 - t2 +1)  ]
        Imputed value of the time series  in the range [t1,t2]  using index_name
     
    
    """
    # map the two boundary points to their sub models
    T_ts = T//no_ts
    m1 = int( max((t1) / int(T_ts / 2) - 1, 0))
    m2 = int( max((t2) / int(T_ts / 2) - 1, 0))
    # query the sub-models parameters
    result = interface.query_table( index_name+'_m',['L', 'start', 'N'], 'modelno =' + str(m1) +' or modelno =' + str(m2)+' order by modelno')
    N1, start1, M1 = result[0]
    # if sub-models are different, get the other sub-model's parameters
    if m1 != m2: N2, start2, M2 = result[1]
    else: N2, start2, M2 = result[0]
    
    # query normalization constants
    col_norm_mean = 'norm_mean'
    col_norm_std = 'norm_std' 
    norm = interface.query_table( index_name+'_m',[col_norm_mean, col_norm_std], 'modelno >=' + str(m1) +' and modelno <=' + str(m2+1)+' order by modelno')

    # Remove when the model writing is fixed (It should write integers directly)
    start1, start2,N1, N2, M1, M2 =  map(int, [start1, start2,N1, N2, M1, M2])
    
    # caluculate tsrow and tscolumn
    if m2 == last_model:
        tscol2 = int((t2 - start2//no_ts) / N2)*no_ts + int((start2)/L) + value_index
        tsrow2 = (t2 - start2//no_ts) % N2
    
    else:
        tscol2 = (int(t2/(N2)))*no_ts + value_index
        tsrow2 = t2 % N2

    if m1 == last_model:
        tscol1 = int((t1 - start1//no_ts) / N1)*no_ts + int((start1)/L) + value_index
        tsrow1 = int((t1- start1//no_ts) % N1)
    
    else:
        tscol1 = (int(t1/(N1)))*no_ts + value_index
        tsrow1 = int(t1 % N1)
    # if tscol are the same
    if tscol1 == tscol2:
        ## change to SUV
        S = interface.get_S_row(index_name + '_s', [m1, m2 + 1], k,
                                         return_modelno=True)
        U = interface.get_U_row(index_name + '_u', [tsrow1, tsrow2], [m1, m2 + 1], k,
                                         return_modelno=True)
        V = interface.get_V_row(index_name + '_v', [tscol1, tscol2], k, value_index,
                                         [m1, m2 + 1],
                                         return_modelno=True)
        mat = np.dot(U[U[:, 0] == m1, 1:] * S[0, 1:], V[V[:, 0] == m1, 1:].T)
        if (m2 < last_model-1 and m1 != 0):
            Result = 0.5 * unnormalize(mat.T.flatten()/p,norm[0][0][value_index],norm[0][1][value_index]) + 0.5 * unnormalize(np.dot(U[U[:, 0] == m1 + 1, 1:] * S[1, 1:],V[V[:, 0] == m1 + 1, 1:].T).T.flatten()/p, norm[1][0][value_index],norm[1][1][value_index])
        else:
            Result = unnormalize(mat.T.flatten()/p,norm[0][0][value_index],norm[0][1][value_index])
        return Result

    else:
        i_index = (t1 - (t1 - start1//no_ts) % N1)
        end = -N2 + tsrow2 + 1
        # Determine returned array size
        
        Result = np.zeros([t2 + - end  - i_index + 1])
        Count = np.zeros(Result.shape)
        # query relevant tuples
        ## change to SUV
        T_e = T//no_ts
        S = interface.get_S_row(index_name + '_s', [m1, m2 + 1], k,
                                         return_modelno=True)
        U = interface.get_U_row(index_name + '_u', [0, 2 * L], [m1, m2 + 1], k,
                                         return_modelno=True)
        V = interface.get_V_row(index_name + '_v', [tscol1, tscol2], k,value_index,
                                         [m1, m2 + 1],
                                         return_modelno=True)
        for m in range(m1, m2 + 1 + (m2 < last_model - 1)):
            mat = np.dot(U[U[:, 0] == m, 1:] * S[m - m1, 1:], V[V[:, 0] == m, 1:].T)
            start = start1//no_ts + int(T_e/2)*(m-m1)
            N = N1
            M = M1
            if m == m2:
                N = N2
                M = M2
            if m == last_model:
                finish = t2 - end +1
                if m1 == last_model:
                    i = 0
                    res = mat.T.flatten()
                    cursor = i_index
                    length = finish - cursor
                else:
                    cursor = max(start + int(T_e/2), i_index)
                    res = mat.T.flatten()
                    i = cursor - i_index
                    i *= (i > 0)
                    length = finish - cursor
                    res = res[-length:]

            else:
                cursor = max(start, i_index)
                finish = (start + M * N//no_ts)
                res = mat.T.flatten()
                i = cursor - i_index
                i *= (i>0)
                length = finish - cursor
            Result[i:i + length] += 0.5 * unnormalize(res/p, norm[m-m1][0][value_index], norm[m-m1][1][value_index])
            Count [i:i + length] += 1
        Result[Count == 1] *= 2
        if end == 0: end = None
        return Result[tsrow1:end]


def _get_forecast_range(index_name,table_name, value_column, index_col, interface, t1, t2,MUpdateIndex,L,k,T,last_model, interval, start_ts, last_TS_seen,no_ts, value_index,direct_var = False,variance = False,averaging = 'average', projected = False,p = 1.0):
    """
    Return the forecasted value in the past at the time range t1 to t2 for the value of column_name using index_name 
    ----------
    Parameters
    ----------
    index_name: string 
        name of the PINDEX used to query the prediction

    index_name: table_name 
        name of the time series table in the database

    value_column: string
        name of column than contain time series value

    index_col: string  
        name of column that contains time series index/timestamp

    interface: db_class object
        object used to communicate with the DB. see ../database/db_class for the abstract class
    
    t1: (int or timestamp)
        index or timestamp indicating the start of the queried range 
    
    t2: (int or timestamp)
        index or timestamp indicating the end of the queried range  
    
    L: (int)
        Model parameter determining the number of rows in each matrix in a sub model. 
    
    k: (int )
        Model parameter determining the number of retained singular values in each matrix in a sub model. 
    
    T: (int )
        Model parameter determining the number of datapoints in each matrix in a sub model.
    
    last_model: (int )
        The index of the last sub model

    averaging: string, optional, (default 'average')
        Coefficients used when forecasting, 'average' means use the average of all sub models coeffcients. 
    ----------
    Returns
    ----------
    prediction  array, shape [(t1 - t2 +1)  ]
        forecasted value of the time series  in the range [t1,t2]  using index_name
    """
    ############### EDITS ##################
    #1- Replace last_ts with the last time stamp seen 
    ########################################
    # get coefficients
    coeffs = np.array(interface.get_coeff(index_name + '_c_view', averaging))
    coeffs_ts = coeffs[-no_ts:]
    coeffs = coeffs[:-no_ts]
    no_coeff = len(coeffs)
 
    if not direct_var or not variance:
            if projected:
                if last_model != 0:
                    q_model = last_model- 1
                else:
                    q_model = last_model
                U = interface.get_U_row(index_name + '_u', [0, 2 * L], [q_model, q_model], k,
                                             return_modelno=False,return_weights_decom=True)[:-1,k:]
                no_coeff = U.shape[0]
                projection_matrix = np.dot(U,U.T)
            
            agg_interval = float(interval)
            if not isinstance(start_ts, (int, np.integer)):
                start_ts = pd.Timestamp(start_ts)
            # if the range queries is beyond what we have so far, get the last point seen
            last_TS_seen = get_bound_time(interface, table_name, index_col, 'max')
            if not isinstance(last_TS_seen, (int, np.integer)):
                last_TS_seen = index_ts_mapper(start_ts, agg_interval, last_TS_seen)
            last_TS_seen+=1
            
            t1_ = min(t1, last_TS_seen)
            t2_ = min(t2, last_TS_seen)
            end = index_ts_inv_mapper(start_ts, agg_interval, t1_ - 1 )
            start = index_ts_inv_mapper(start_ts, agg_interval, t1_ - no_coeff  )
            obs = interface.get_time_series(table_name, start, end, start_ts = start_ts,  value_column=value_column, index_column= index_col, Desc=False, interval = agg_interval, aggregation_method =  averaging)
            output = np.zeros([t2 - t1_ + 1 ])
            obs = np.array(obs)[-no_coeff:,0]
            # Fill using fill_method
            if p <1:
                obs = np.array(pd.DataFrame(obs).fillna(value = 0).values[:,0])
                obs /= p
            else:
                obs = np.array(pd.DataFrame(obs).fillna(method = 'ffill').values[:,0])
                obs = np.array(pd.DataFrame(obs).fillna(method = 'bfill').values[:,0])
            if variance:
                obs = obs **2
            observations = np.zeros([t2 - t1_ + 1 + no_coeff])
            observations[:no_coeff] = obs
            
            for i in range(0, t2 + 1 - t1_): 
                    if i  < len(obs):
                        if projected:
                            output[i] = np.dot(coeffs.T, np.dot(projection_matrix, observations[i:i + no_coeff]))+coeffs_ts[value_index]
                        else:
                            output[i] = np.dot(coeffs.T,  observations[i:i + no_coeff])+coeffs_ts[value_index]
                    else:
                        output[i] = np.dot(coeffs.T,  observations[i:i + no_coeff])+coeffs_ts[value_index]
                    if i+no_coeff >= len(obs):
                        observations[i+no_coeff] = output[i]

            return output[-(t2 - t1 + 1):]
            
    # the forecast should always start at the last point
    t1_ = MUpdateIndex//no_ts 
    output = np.zeros([t2 - t1_ + 1 + no_coeff])
    output[:no_coeff] = _get_imputation_range(index_name, table_name, value_column, index_col, interface, t1_ - no_coeff, t1_ - 1, L,k,T,last_model,value_index, no_ts)
    for i in range(0, t2 + 1 - t1_):
        output[i + no_coeff] = np.dot(coeffs.T, output[i:i + no_coeff])+coeffs_ts[value_index]
    return output[-(t2 - t1 + 1):]
    


def _get_imputation(index_name, table_name, value_column, index_col, interface, t,L,k,T,last_model, no_ts,value_index, p = 1.0):
    """
    Return the imputed value in the past at time t for the value of column_name using index_name 
        ----------
        Parameters
        ----------
        index_name: string 
            name of the PINDEX used to query the prediction

        index_name: table_name 
            name of the time series table in the database

        value_column: string
            name of column than contain time series value

        index_col: string  
            name of column that contains time series index/timestamp

        interface: db_class object
            object used to communicate with the DB. see ../database/db_class for the abstract class
        
        t: (int or timestamp)
            index or timestamp indicating the queried time. 
        
        L: (int)
            Model parameter determining the number of rows in each matrix in a sub model. 
        
        k: (int )
            Model parameter determining the number of retained singular values in each matrix in a sub model. 
        
        T: (int or timestamp)
            Model parameter determining the number of datapoints in each matrix in a sub model.
        
        last_model: (int or timestamp)
            The index of the last sub model
        ----------
        Returns
        ----------
        prediction float
            Imputed value of the time series at time t using index_name
         
        
    """
    # map t to the right sub model
    T_ts = T//no_ts
    modelNo = int( max((t) / int(T_ts / 2) - 1, 0))
    N = L
    col_norm_mean = 'norm_mean' 
    col_norm_std = 'norm_std'
    norm = interface.query_table( index_name+'_m',[col_norm_mean, col_norm_std], 'modelno =' + str(modelNo) +' or modelno =' + str(modelNo+1)+' order by modelno')
    # if it is in the last sub-model, tscol and tsrow will be calculated differently
    if modelNo == last_model:

        N, last_model_start = interface.query_table( index_name+'_m',['L', 'start'], ' modelno =' + str(modelNo) )[0]
        tscolumn = int((t - last_model_start//no_ts) / N)*no_ts + value_index + int((last_model_start)/L)
        tsrow = (t - last_model_start//no_ts) % N
        U, S, V = interface.get_SUV(index_name, [tscolumn, tscolumn], [tsrow, tsrow],
                                            [modelNo, modelNo + 1], k, value_index)
        return unnormalize(sum([a * b * c for a, b, c in zip(U[0, :], S[0, :], V[0, :])])/p, norm[0][0][value_index], norm[0][1][value_index])
        # U

    # if it is in the model before last, do not query the last model
    elif modelNo == last_model - 1:
        tscolumn = (int(t/(N)))*no_ts + value_index
        tsrow = t % N
        U, V, S = interface.get_SUV(index_name, [tscolumn, tscolumn], [tsrow, tsrow],
                                            [modelNo, modelNo], k,value_index)
        return unnormalize(sum([a * b * c for a, b, c in zip(U[0, :], S[0, :], V[0, :])])/p, norm[0][0][value_index], norm[0][1][value_index])
    
    else:
        tscolumn = (int(t/(N)))*no_ts + value_index
        tsrow = t % N
        U, S, V = interface.get_SUV(index_name, [tscolumn, tscolumn], [tsrow, tsrow],
                                            [modelNo, modelNo + 1],k,value_index)
  
        # if two sub models are queried get the average
        if V.shape[0] == 2 and U.shape[0] == 2:
            return 0.5* (unnormalize(np.sum(U[0,:] * S[0] * V[0,:])/p, norm[0][0][value_index],norm[0][1][value_index])+ unnormalize(np.sum(U[1,:] * S[1] * V[1,:])/p, norm[1][0][value_index],norm[1][1][value_index]))
            #return 0.5 * (sum([a * b * c for a, b, c in zip(U[0, :], S[0, :], V[0, :])]) + sum(
            #    [a * b * c for a, b, c in zip(U[1, :], S[1, :], V[1, :])]))
        
        # else return one value directly
        return unnormalize(sum([a * b * c for a, b, c in zip(U[0, :], S[0, :], V[0, :])])/p,  norm[0][0][value_index], norm[0][1][value_index])

def forecast_next(index_name,table_name, value_column, index_col, interface, averaging = 'last1', ahead = 1):
    """
    Return the florcasted value in the past at the time range t1 to t2 for the value of column_name using index_name 
    ----------
    Parameters
    ----------
    index_name: string 
        name of the PINDEX used to query the prediction

    index_name: table_name 
        name of the time series table in the database

    value_column: string
        name of column than contain time series value

    index_col: string  
        name of column that contains time series index/timestamp

    interface: db_class object
        object used to communicate with the DB. see ../database/db_class for the abstract class
    
    averaging: string, optional, (default 'average')
        Coefficients used when forecasting, 'average' means use the average of all sub models coeffcients. 
    ----------
    Returns
    ----------
    prediction  array, shape [(t1 - t2 +1)  ]
        forecasted value of the time series  in the range [t1,t2]  using index_name
    """
    # get coefficients
    coeffs = np.array(interface.get_coeff(index_name + '_c_view', averaging))
    no_coeff = len(coeffs)
    # get parameters
    end_index , agg_interval, start_ts = interface.query_table( index_name+'_meta',["last_TS_seen", 'agg_interval','start_time'])[0]
    agg_interval = float(agg_interval)
    
    if not isinstance(start_ts, (int, np.integer)):
        start_ts = pd.Timestamp(start_ts)
     
    end = index_ts_inv_mapper(start_ts, agg_interval, end_index)
    start = index_ts_inv_mapper(start_ts, agg_interval, end_index-no_coeff )
    # the forecast should always start at the last point
    obs = interface.get_time_series( table_name, start, end, start_ts = start_ts,  value_column=value_column, index_column= index_col, Desc=False, interval = agg_interval, aggregation_method = averaging)
    output = np.zeros(ahead+no_coeff)
    output[:no_coeff] = np.array(obs)[:,0]
    for i in range(0, ahead):
        output[i + no_coeff] = np.dot(coeffs.T, output[i:i + no_coeff])
    return output[-ahead:]



