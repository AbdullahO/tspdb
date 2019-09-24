
import numpy as np
from tspdb.src.database_module.sql_imp import SqlImplementation
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex
from tspdb.src.pindex.pindex_utils import  index_ts_mapper
import time
interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
import timeit
import pandas as pd
# t = time.time()
# A = load_pindex(interface, 'ts_basic2_pindex')
# print (time.time() - t)
# print(' ')
# t = time.time()
# a = interface.get_time_series('ts_basic2',1,10**6, index_col = 'row_id')
# print (time.time() - t)
# print(' ')
def update_test(init_points = 10**4 , update_points = [1000,100,5000,10000], T = 1000, direct_var = True ,index_name = 'ts_basic_test_pindex'):
	interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
	df = pd.DataFrame(data ={'ts': np.arange(init_points).astype('float')}) 
	interface.create_table('ts_basic_test', df, 'row_id', index_label='row_id')
	time_series_table = ['ts_basic_test','ts', 'row_id']
	T0 = 1000
	gamma = 0.5
	k = 2
	k_var = 1
	TSPD = TSPI(_dir = 'C:/Program Files/PostgreSQL/10/data/', agg_interval = 5, T = T,T_var = T, rank = k, rank_var =  k_var, col_to_row_ratio = 10, index_name = index_name,gamma = gamma, interface= interface ,time_series_table = time_series_table, direct_var = direct_var )
	TSPD.create_index()
	interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
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

	# start = pd.to_datetime('2015-01-01 00:00:00')
	# end = pd.to_datetime('2018-01-01 00:00:00')
	# df = pd.DataFrame(data ={'ts': np.arange(init_points).astype('float')}) 
	# timestamps= pd.DatetimeIndex(1000000000.*(np.random.randint(start.value/10**9,end.value/10**9, init_points)))
	# df.index = np.sort(timestamps)
	# interface.create_table('ts_basic_ts2', df, 'timestamp', index_label='timestamp')

	
def create_pindex_test(table_name, T,T_var, k ,k_var, direct_var, index_name = None , agg_interval = 1., col_to_row_ratio= 10, time_column = 'row_id'):

	interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
	time_series_table = [table_name,'ts', time_column]
	T0 = 1000
	gamma = 0.5
	TSPD = TSPI(T = T,T_var = T, rank = k, rank_var =  k_var,agg_interval=agg_interval, col_to_row_ratio = col_to_row_ratio, index_name = index_name,gamma = gamma, interface= interface ,time_series_table = time_series_table, direct_var = direct_var )
	TSPD.create_index()
	print(TSPD.ts_model._denoiseTS()[-10:], TSPD.ts_model.TimeSeriesIndex)

def range_prediction_queries_test(index_name, table_name, max_):
	
	T1 = [0,0,max_-10, max_-15000, max_] + list((max_+1000) * np.random.random(10))
	T2 = [10, 10**5, max_-1, max_, max_ +10] + list((max_+1000) * np.random.random(10))
	T1 = np.array(T1).astype(int)
	T2 = np.array(T2).astype(int)
	for t1_,t2_ in zip(T1,T2):
		t1,t2 = sorted([t1_,t2_])
		# print (')
		# try:
		get_prediction_range(index_name, table_name, 'ts', 'row_id', interface, int(t1),int(t2), uq = True)
		# except: print('failure to query  range between %s and %s' % (t1,t2))

def prediction_queries_test(index_name, table_name, max_):
	
	T1 = [0,max_-10, max_-1000, max_+1, max_+10] + list((max_+1000) * np.random.random(50))
	T1 = np.array(T1).astype(int)

	for t1 in T1:
		# try: 
			get_prediction(index_name, table_name, 'ts', 'row_id', interface, int(t1), uq = True)
		# except: print('failure to query  point  %s' %t1)

def prediction_queries_accuracy_test(max_, index_name = "tspdb.ts_basic2_pindex2", table_name = "ts_basic2"):
	T1 = [100000,max_-1000, max_] + list((max_-1) * np.random.random(100))
	T1 = np.array(T1).astype(int)
	
	for t1 in T1:
		print ('t = '+str(t1))
		A,_ = get_prediction(index_name, table_name, 'ts', 'row_id', interface, int(t1))
		print (t1,A )
		assert abs(A - t1) < 1e-3
	

def range_prediction_queries_accuracy_test(max_, index_name = "tspdb.ts_basic2_pindex2", table_name = "ts_basic2"):
	T1 = [0,0,max_-10, max_-1000] + list((max_-1) * np.random.random(10))
	T2 = [10, max_, max_-1, max_] + list((max_-1) * np.random.random(10))
	T1 = np.array(T1).astype(int)
	T2 = np.array(T2).astype(int)
	
	for t1_,t2_ in zip(T1,T2):
		t1,t2 = sorted([t1_,t2_])
		A,_ = get_prediction_range(index_name, table_name, 'ts', 'row_id', interface, int(t1),int(t2), uq = True)
		print (t1,t2,np.max(A - np.arange(t1,t2+1)))
		print(A)
		assert abs(np.max(A - np.arange(t1,t2+1))) < 1e-3


def prediction_queries_latency_test():
	setup = '''import numpy as np
from tspdb.src.database_module.sql_imp import SqlImplementation
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
interface = SqlImplementation(driver="postgresql", host="localhost", database="querytime_test",user="aalomar",password="AAmit32lids")
	'''
	stmt1 = "get_prediction('tspdb.ts_5_pindex', 'ts_5', 'ts', 'row_id', interface, 10,False)"
	stmt2 = "get_prediction('tspdb.ts_basic2_pindex', 'ts_basic2', 'ts', 'row_id', interface, 10,False)"
	stmt3 = "get_prediction('tspdb.ts_5_pindex_2', 'ts_5', 'ts', 'row_id', interface, 10,False)"
	
	
	stmtA = '''interface.execute_query("select ts from ts_basic2 where row_id = 10") '''
	stmtB = '''interface.execute_query("select ts from ts_5 where row_id = 390") '''
	
	print ('(test1 pindex: ts_5_pindex )	  	imp query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt1, number =1000)/timeit.timeit(setup = setup,stmt= stmtB, number =1000)))
	print ('(test2 pindex: ts_basic2_pindex ) 	imp query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt2, number =1000)/timeit.timeit(setup = setup,stmt= stmtA, number =1000)))
	print ('(test3 pindex: ts_5_pindex_2 )		imp query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt3, number =1000)/timeit.timeit(setup = setup,stmt= stmtB, number =1000)))

	stmt1 = "get_prediction('tspdb.ts_5_pindex', 'ts_5', 'ts', 'row_id', interface, 99997  ,False)"
	stmt2 = "get_prediction('tspdb.ts_basic2_pindex', 'ts_basic2', 'ts', 'row_id', interface, 999824 ,False)"
	stmt3 = "get_prediction('tspdb.ts_5_pindex_2', 'ts_5', 'ts', 'row_id', interface, 99970  ,False)"
	
	print ('(test1 pindex: ts_5_pindex )	  	Forecast query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt1, number =1000)/timeit.timeit(setup = setup,stmt= stmtB, number =1000)))
	print ('(test2 pindex: ts_basic2_pindex ) 	Forecast query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt2, number =1000)/timeit.timeit(setup = setup,stmt= stmtA, number =1000)))
	print ('(test3 pindex: ts_5_pindex_2 )		Forecast query latency is %s that of SELECT ' %(timeit.timeit(setup = setup,stmt= stmt3, number =1000)/timeit.timeit(setup = setup,stmt= stmtB, number =1000)))


def main():
	# ts_table_tests()
	# print('test 1')

	update_test()
	
	print('test 2')
	update_test(init_points = 10**4 , update_points = [10**4, 100,1,100000,10002], T = 10000, index_name = 'haha')
	
	# print('test 3')
	# update_test(init_points = 10**4 , update_points = [10**4], T = 1000000, index_name = 'hahddda')
	
	# print('test 4')
	# update_test(init_points = 10**4 , T = 1000000, direct_var = False, index_name = 'hahaeee')
	# t2 = time.time()
	# create_pindex_test('ts_basic2', 250000,250000, 2 ,1, True, index_name = 'ts_basic2_pindex', time_column = 'row_id',agg_interval = 1 )
	# print ('create pindex test: basic test, time = %s seconds, %s record/s'%(time.time()-t2, (time.time()-t2)/10**6))
	# print('=======================================================')
	# print('SUCCESS')
	# print('=======================================================')
	
	# t2 = time.time()
	# create_pindex_test('ts_basic2', 30000,30000, 2 ,1, True, index_name = 'ts_basic2_pindex2', time_column = 'row_id',agg_interval = 1 )
	# print ('create pindex test: basic test T = 10000, time = %s seconds, %s record/s'%(time.time()-t2, (time.time()-t2)/10**6))
	# print('=======================================================')
	# print('SUCCESS')
	# print('=======================================================')
	
	# t2 = time.time()
	# create_pindex_test('ts_5', 9240,9240, 2 ,1, True, index_name = 'ts_5_pindex' )
	# print ('create pindex test: T = 9240, time = %s seconds, %s record/s' %(time.time()-t2, (time.time()-t2)/10**5))
	# print('=======================================================')
	# print('SUCCESS')
	# print('=======================================================')
	
	# t2 = time.time()
	# create_pindex_test('ts_5', 10000,10000, 2 ,1, False, index_name = 'ts_5_pindex_2' , col_to_row_ratio = 1)
	# print ('create pindex test: non-direct_var, time = %s seconds,  %s record/s'%(time.time()-t2, (time.time()-t2)/10**5))
	
	# print('testing 15 range queries in index ts_basic2_pindex')
	# range_prediction_queries_test('tspdb.ts_basic2_pindex', 'ts_basic2', 10**6)
	# print('=======================================================')
	# print('SUCCESS')

	# print('=======================================================')
	# print('testing 50 point queries in index ts_basic2_pindex')
	# prediction_queries_test('tspdb.ts_basic2_pindex', 'ts_basic2', 10**6)
	# print('=======================================================')
	# print('SUCCESS')
	
	# print('=======================================================')
	# print('testing 15 range queries in index ts_basic2_pindex2')
	# range_prediction_queries_test('tspdb.ts_basic2_pindex2', 'ts_basic2', 10**6)
	# print('=======================================================')
	# print('SUCCESS')

	# print('=======================================================')
	# print('testing 50 point queries in index ts_basic2_pindex2')
	# prediction_queries_test('tspdb.ts_basic2_pindex2', 'ts_basic2', 10**6)
	# print('=======================================================')
	# print('SUCCESS')
	
	# print('=======================================================')
	# print('testing 15 range queries in index ts_5_pindex')
	# range_prediction_queries_test('tspdb.ts_5_pindex', 'ts_5', 10**5)
	# print('=======================================================')
	# print('SUCCESS')
	# print('=======================================================')
	
	# print('testing 50 point queries in index ts_5_pindex')
	# prediction_queries_test('tspdb.ts_5_pindex', 'ts_5', 10**5)
	# print('=======================================================')
	# print('SUCCESS')
	
	# print('=======================================================')
	# print('testing 15 range queries in index ts_5_pindex_2')
	# range_prediction_queries_test('tspdb.ts_5_pindex_2', 'ts_5', 10**5)
	# print('=======================================================')
	# print('SUCCESS')
	
	# print('=======================================================')
	# print('testing 50 point queries in index ts_5_pindex_2')
	# prediction_queries_test('tspdb.ts_5_pindex_2', 'ts_5', 10**5)
	# print('=======================================================')
	# print('SUCCESS')
	
	# prediction_queries_latency_test()
	# print('=======================================================')
	# print('Imputation accuracy Test')
	# range_prediction_queries_accuracy_test(999824)
	# prediction_queries_accuracy_test(999824)
	# print('=======================================================')
	# print('SUCCESS')

main()
# max_ = 99995
# T1 = (max_ * np.random.random(100)).astype(int)
# T2 = (max_ * np.random.random(100)).astype(int)
# for t1_,t2_ in zip(T1,T2):
# 		t1,t2 = sorted([t1_,t2_])
# 		print(get_prediction_range('ts_5_pindex', 'ts_5', 'ts', 'row_id', interface, int(t1),int(t2), uq = False))

# get_prediction_range('ts_5_pindex', 'ts_5', 'ts', 'row_id', interface, 1,6, uq = False)
# stmt1 = "get_prediction_range('ts_5_pindex', 'ts_basic2', 'ts', 'row_id', interface, 10,0)"
# timeit.timeit(setup = setup,stmt= stmt1, number =1000)

# stmt = '''interface.engine.execute("select prediction from predict('ts_basic2', 'ts', 'row_id', 10, c=>95,  uq => false);")'''
# timeit.timeit(setup = setup,stmt= stmt, number =1000)

# stmt2 = '''interface.engine.execute("select ts from ts_basic2 where row_id = 10") '''
# timeit.timeit(setup = setup,stmt= stmt2, number =1000)

# timeit.timeit(setup = setup,stmt= stmt, number =1000)/timeit.timeit(setup = setup,stmt= stmt2, number =1000)

# stmt = '''interface.engine.execute("select prediction from predict('ts_basic2', 'ts', 'row_id', 1000000, c=>95,  uq => false);")'''
# timeit.timeit(setup = setup,stmt= stmt, number =1000)


# timeit.timeit(setup = setup,stmt= stmt, number =1000)/timeit.timeit(setup = setup,stmt= stmt2, number =1000)

# stmt = '''interface.engine.execute("select prediction from predict('ts_basic2', 'ts', 'row_id', 1000000, c=>95,  uq => true);")'''
# timeit.timeit(setup = setup,stmt= stmt, number =1000)

# timeit.timeit(setup = setup,stmt= stmt, number =1000)/timeit.timeit(setup = setup,stmt= stmt2, number =1000)


# stmt = '''interface.engine.execute("select prediction from predict('ts_basic2', 'ts', 'row_id', 10, c=>95,  uq => true);")'''
# timeit.timeit(setup = setup,stmt= stmt, number =1000)

# timeit.timeit(setup = setup,stmt= stmt, number =1000)/timeit.timeit(setup = setup,stmt= stmt2, number =1000)

# coeff =  [i[0] for i in interface.engine.execute('select coeffvalue from ts_5_pindex_c where modelno = 1 order by coeffpos;').fetchall()]
# t = 100000
# ouput =  [i[0] for i in interface.engine.execute('select ts from ts_5 where row_id<= 100000-30;').fetchall()]
