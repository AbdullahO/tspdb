
import numpy as np
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI
from tspdb.src.pindex.pindex_utils import  index_ts_mapper
import time
import timeit
import pandas as pd
from tspdb.src.hdf_util import read_data
from tspdb.src.tsUtils import randomlyHideValues
from scipy.stats import norm
from sklearn.metrics import r2_score
import tspdb

def r2_var(y,y_h,X):
    average = np.mean(X**2) - np.mean(X)**2
    return 1 - sum((y-y_h)**2)/sum((y-average)**2)

def create_table_data():

	obs = np.arange(10**5).astype('float')
	means = obs
	var = np.zeros(obs.shape)
	obs_9 = randomlyHideValues(np.array(obs), 0.9)[0]
	obs_7 = randomlyHideValues(np.array(obs), 0.7)[0]
	print(obs_9)
	df = pd.DataFrame(data ={'ts':obs, 'means':  means, 'ts_9':obs_9, 'ts_7' : obs_7,'var': var  }) 
	df.to_csv('testdata/tables/ts_basic_5.csv',index_label = 'time')

	timestamps = pd.date_range('2012-10-01 00:00:00', periods = 10**5, freq='5s')
	df.index = timestamps
	df.to_csv('testdata/tables/ts_basic_ts_5_5.csv', index_label = 'time')
	

	# real time series variance constant
	data = read_data('testdata/MixtureTS2.h5')
	obs = data['obs'][:]
	means = data['means'][:]
	var = np.ones(obs.shape)
	obs_9 = randomlyHideValues(np.array(obs), 0.9)[0]
	obs_7 = randomlyHideValues(np.array(obs), 0.7)[0]
	df = pd.DataFrame(data ={'ts':obs, 'means':  means, 'ts_9':obs_9, 'ts_7' : obs_7 ,'var': var }) 
	df.index_label = 'time'
	df.to_csv('testdata/tables/MixtureTS2.csv', index_label = 'time')
	
	# real time series variance constant
	data = read_data('testdata/MixtureTS.h5')
	obs = data['obs'][:]
	means = data['means'][:]
	var = np.ones(obs.shape)
	obs_9 = randomlyHideValues(np.array(obs), 0.9)[0]
	obs_7 = randomlyHideValues(np.array(obs), 0.7)[0]
	df = pd.DataFrame(data ={'ts':obs, 'means':  means, 'ts_9':obs_9, 'ts_7' : obs_7,'var': var  }) 
	df.to_csv('testdata/tables/MixtureTS.csv', index_label = 'time')
	
	# real time series varaince harmonics
	data = read_data('testdata/MixtureTS_var.h5')
	obs = data['obs'][:]
	means = data['means'][:]
	var = data['var'][:]
	obs_9 = randomlyHideValues(np.array(obs), 0.9)[0]
	obs_7 = randomlyHideValues(np.array(obs), 0.7)[0]
	df = pd.DataFrame(data ={'ts':obs, 'means':  means, 'ts_9':obs_9, 'ts_7' : obs_7, 'var': var  }) 
	df.to_csv('testdata/tables/MixtureTS_var.csv', index_label = 'time')

def create_tables(interface):
	dir_ = tspdb.__path__[0]+'/tests/'	
	for table in ['mixturets2','ts_basic_5','ts_basic_ts_5_5','mixturets_var']:
		df = pd.read_csv(dir_+'testdata/tables/%s.csv'%table, engine = 'python') 
		if table == 'ts_basic_ts_5_5': df['time'] = df['time'].astype('datetime64[ns]')
		interface.create_table(table, df, 'time', include_index = False)
	

def update_test(interface, init_points = 10**4 , update_points = [1000,100,5000,10000], T = 1000, direct_var = True ,index_name = 'ts_basic_test_pindex'):
	df = pd.DataFrame(data ={'ts': np.arange(init_points).astype('float')}) 
	interface.create_table('ts_basic_test', df, 'row_id', index_label='row_id')
	time_series_table = ['ts_basic_test','ts', 'row_id']
	T0 = 1000
	gamma = 0.5
	k = 2
	k_var = 1
	agg_interval = 1.
	conn = interface.engine.raw_connection()
	cur = conn.cursor()
	cur.execute('''SELECT create_pindex('%s','%s','%s','%s', "T" => %s, k => %s, k_var => %s, agg_interval => %s, var_direct => %s)'''%('ts_basic_test','row_id','ts', index_name, T, k,k_var, agg_interval, direct_var))
	cur.close()
	conn.commit()
	conn.close()
	for points in update_points:
		df = pd.DataFrame(data = {'ts':np.arange(init_points,points+init_points).astype('float')}, index = np.arange(init_points,points+init_points) ) 
		interface.bulk_insert('ts_basic_test', df, index_label='row_id')
		init_points += points
		print ('successfully updated %s points' %points)
		
def ts_table_tests(init_points = 10**4 , update_points = [1000,100,5000,10000], T = 1000, direct_var = True ,index_name = 'ts_basic_ts_pindex'):
	interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
	
	df = pd.DataFrame(data ={'ts': np.arange(init_points).astype('float')}) 
	timestamps = pd.date_range('2012-10-01 00:00:00', periods = init_points+1, freq='5s')
	end = timestamps[-1]
	df.index = timestamps[:-1]
	interface.create_table('ts_basic_ts', df, 'timestamp', index_label='timestamp')
	time_series_table = ['ts_basic_ts','ts', 'timestamp']
	T0 = 1000
	gamma = 0.5
	k = 2
	k_var = 1
	TSPD = TSPI(_dir = 'C:/Program Files/PostgreSQL/10/data/', agg_interval = 5, T = T,T_var = T, rank = k, rank_var =  k_var, col_to_row_ratio = 10, index_name = index_name,gamma = gamma, interface= interface ,time_series_table = time_series_table, direct_var = direct_var )
	TSPD.create_index()
	interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
	for points in update_points:
		df = pd.DataFrame(data = {'ts':np.arange(init_points,points+init_points).astype('float')} ) 
		timestamps = pd.date_range(end, periods = points+1, freq='5s')
		end = timestamps[-1]
		df.index = timestamps[:-1]
		interface.bulk_insert('ts_basic_ts', df, index_label='timestamp')
		init_points += points
		print ('successfully updated %s points' %points)

	
def create_pindex_test(interface,table_name, T,T_var, k ,k_var, direct_var,value_column= ['ts'], index_name = None , agg_interval = 1., col_to_row_ratio= 10, time_column = 'row_id'):

	T0 = 1000
	gamma = 0.5
	if index_name is None: index_name = 'pindex'
	value_column = ','.join(value_column)
	interface.engine.execute('''SELECT create_pindex('%s','%s','{%s}','%s', T => %s,t_var =>%s, k => %s, k_var => %s, agg_interval => %s, var_direct => %s, col_to_row_ratio => %s)'''%(table_name,time_column, value_column, index_name, T, T_var, k,k_var, agg_interval, direct_var, col_to_row_ratio))


