import sys
import numpy as np

from tspdb.src.pindex.pindex_managment import TSPI
from tspdb.src.database_module.sql_imp import SqlImplementation
from tspdb.src.database_module.plpy_imp import plpyimp
from sklearn.metrics import r2_score
from time import clock
import pandas as pd

T0 = 1000
T = 2500000
gamma = 0.5
k = 2
time_series_table = ['mixturets2','ts','time']
TSPD = TSPI(T = T, rank = k, interface= SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",
               user="postgres",password="0505") ,time_series_table = time_series_table, recreate = True)
TSPD.create_index()
df = pd.DataFrame(data= {'ts': np.arange(10**6)})
TSPD.db_interface.create_table('ts_basic2', df, include_index = True)
TSPD.update_model(np.arange(1,10**7+1))
TSPD.write_model()




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
