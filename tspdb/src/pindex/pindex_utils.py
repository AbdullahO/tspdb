import numpy as np
import pandas as pd
from dateutil.parser import parse

def index_ts_mapper(start, interval, timestamp):
    """
    takes time series index  (timestamp) and return the integer index in model
    """
    if isinstance(start, (int, np.integer)):
        return int((timestamp-start)/(interval))
    
    elif  isinstance(start, (pd.Timestamp)):
        return int((timestamp.value-start.value)/(interval*10**9))
    
    else:
        raise Exception('start value for the mapper must either be integers or pd.timestamp')

def index_ts_inv_mapper(start, interval, index):
    """
    takes integer index in model  (index) and return the time series index
    """
    if isinstance(start, (int, np.integer)):
        return int((index *interval) + start)

    elif  isinstance(start, (pd.Timestamp)):
        return  pd.to_datetime(float(index*(interval*10**9)+start.tz_localize(None).value))

    else:
        raise Exception('start value for the inv_mapper must either be integers or pd.timestamp')
def index_exists(interface, index_name ):
    """
    :return:
    """
    return interface.table_exists(index_name+'_meta')

def get_bound_time(interface, time_series_table, time_column, exterme = 'min'):
    min_ = interface.get_extreme_value(time_series_table, time_column, exterme)
    if isinstance(min_, (int, np.integer)): return min_
    
    else: 
        min_ = parse(min_)
        return pd.to_datetime(min_).tz_localize(None)
