import sys
import numpy as np

from tspdb.src.pindex.pindex_managment import TSPI
from tspdb.src.pindex.predict import get_prediction, get_prediction_range

from tspdb.src.database_module.sql_imp import SqlImplementation
from tspdb.src.database_module.plpy_imp import plpyimp

from sklearn.metrics import r2_score
from time import clock
import pandas as pd

T0 = 1000
T = 90000
gamma = 0.5
k = 2
time_series_table = ['mixturets2','ts','time']
TSPD = TSPI(T = T, rank = k, interface= SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",
               user="postgres",password="0505") ,value_column = ['ts','ts_7','ts_9'],time_series_table_name = time_series_table[0], time_column = time_series_table[2])

# TSPD.update_model(np.arange(1,10**6+1))
# TSPD.write_model(1)


TSPD.create_index()
get_prediction('tspdb.pindex_mixturets2', 'mixturets2', 'ts_9', TSPD.db_interface, 99942)
# df = pd.DataFrame(data= {'ts': np.arange(10**6)})
# TSPD.db_interface.create_table('ts_basic2', df, include_index = True)
# TSPD.update_model(np.arange(1,10**7+1))
# TSPD.write_model()
get_prediction_range('tspdb.pindex_mixturets2', 'mixturets2', 'ts_9', TSPD.db_interface, 0,10)


CREATE or REPLACE FUNCTION test() RETURNS void  AS $$   
try: 1+1 
except: print(1) 
$$LANGUAGE plpython3u

# print 'Done creating index ..'
# model = TSPD.ts_model
# T = (10**7*np.random.random(1000)).astype(int)
# for t in T:
#     t_h = TSPD.get_imputation(t,TSPD.ts_model)
#     assert (np.abs(t_h-t-1) < 1e-3)
#
#
# T = (10**7*np.random.random(1000)).astype(int)
# T2 = (10**7*np.random.random(1000)).astype(int)
# for t1_,t2_ in zip(T,T2):
#     t1 = min(t1_,t2_)
#     t2 = max(t1_,t2_)
#     t_h = TSPD.get_imputation_range(t1,t2,TSPD.ts_model)
#     assert (np.max(np.abs(t_h-np.arange(t1,t2+1)-1)) < 1e-3)
