-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION tspdb" to load this file. \quit

CREATE SCHEMA IF NOT EXISTS  tspdb;

CREATE TABLE IF NOT EXISTS tspdb.pindices (
	index_name text PRIMARY key ,
	relation text  not NULL ,
	time_column text  not NULL ,
	uq boolean  not NULL ,
	agg_interval double precision not NULL ,
	initial_timestamp timestamp,
	last_timestamp timestamp,
	initial_index bigint,
	last_index bigint
);

CREATE TABLE IF NOT EXISTS tspdb.pindices_columns (
	model_id SERIAL PRIMARY key,
	index_name text not NULL,
	value_column text  not NULL );

CREATE or REPLACE FUNCTION create_pindex (table_name text, ts_column text, value_column text, index_name text, t_var int DEFAULT -1 ,k int DEFAULT 3 , k_var int DEFAULT 1, "T" int DEFAULT 2500000, "T0" int DEFAULT 1000,var_direct boolean DEFAULT true,gamma numeric DEFAULT 0.5, col_to_row_ratio int DEFAULT 10, agg_interval numeric DEFAULT 1.0)
RETURNS void AS $$
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex
from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
pass

# Build index 
time_series_table = [table_name,value_column,ts_column]

TSPD = TSPI(T = T,T_var = T, rank = k, rank_var =  k_var, gamma = gamma, col_to_row_ratio = col_to_row_ratio, interface= plpyimp(plpy) ,time_series_table = time_series_table, recreate = True, direct_var = var_direct, index_name = index_name, agg_interval = agg_interval)
TSPD.create_index()


$$ LANGUAGE plpythonu;

CREATE or REPLACE FUNCTION predict (table_name text, value_column text, t int,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,  OUT prediction numeric, OUT LB numeric,OUT UB numeric)
AS $$

from tspdb.src.pindex.predict import get_prediction
from tspdb.src.database_module.plpy_imp import plpyimp
#check if index exist or if there exist index that is implemented for column 

# get 
index_name_ = 'tspdb.'+index_name
if not uq:
  prediction = get_prediction( index_name_, table_name, value_column, '', plpyimp(plpy), t, uq)
  return prediction, 0,0
else: 
  prediction,interval = get_prediction( index_name_, table_name, value_column, '', plpyimp(plpy), t, uq, uq_method = uq_method, c = c)
  return prediction, prediction-interval, prediction+ interval
$$ LANGUAGE plpythonu;

CREATE or REPLACE FUNCTION predict (table_name text, value_column text, t timestamp,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,  OUT prediction numeric, OUT LB numeric,OUT UB numeric)
AS $$

from tspdb.src.pindex.predict import get_prediction
from tspdb.src.database_module.plpy_imp import plpyimp
#check if index exist or if there exist index that is implemented for column 

# get 
index_name_ = 'tspdb.'+index_name
if not uq:
  prediction = get_prediction( index_name_, table_name, value_column, '', plpyimp(plpy), t, uq)
  return prediction, 0,0
else: 
  prediction,interval = get_prediction( index_name_, table_name, value_column, '', plpyimp(plpy), t, uq, uq_method = uq_method, c = c)
  return prediction, prediction-interval, prediction+ interval
$$ LANGUAGE plpythonu;

CREATE or REPLACE FUNCTION predict_range (table_name text, value_column text,  t1 int, t2 int,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95, OUT prediction numeric, OUT LB numeric,OUT UB numeric)
RETURNS SETOF record AS $$
from tspdb.src.pindex.predict import get_prediction_range
from tspdb.src.database_module.plpy_imp import plpyimp

#check if index exist or if there exist index that is implemented for column 
index_name_ = 'tspdb.'+index_name
# get 
if not uq:
  prediction = get_prediction_range( index_name_, table_name, value_column, '', plpyimp(plpy), t1,t2, uq)
  return zip(prediction, prediction,prediction)
else: 
  prediction,interval = get_prediction_range( index_name_, table_name, value_column, '', plpyimp(plpy), t1,t2,uq, uq_method = uq_method, c = c)
  return zip(prediction, prediction-interval, prediction+ interval)
$$ LANGUAGE plpythonu;

CREATE or REPLACE FUNCTION predict_range (table_name text, value_column text,  t1 timestamp, t2 timestamp,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95, OUT prediction numeric, OUT LB numeric,OUT UB numeric)
RETURNS SETOF record AS $$
from tspdb.src.pindex.predict import get_prediction_range
from tspdb.src.database_module.plpy_imp import plpyimp

#check if index exist or if there exist index that is implemented for column 
index_name_ = 'tspdb.'+index_name
# get 
if not uq:
  prediction = get_prediction_range( index_name_, table_name, value_column, '', plpyimp(plpy), t1,t2, uq)
  return zip(prediction, prediction,prediction)
else: 
  prediction,interval = get_prediction_range( index_name_, table_name, value_column, '', plpyimp(plpy), t1,t2,uq, uq_method = uq_method, c = c)
  return zip(prediction, prediction-interval, prediction+ interval)
$$ LANGUAGE plpythonu;


CREATE or REPLACE FUNCTION update_pindex(index_name text)
RETURNS void AS $$
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex
from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
# Build index 
TSPD = load_pindex(plpyimp(plpy),index_name)
TSPD.update_index()
$$ LANGUAGE plpythonu;

CREATE or REPLACE FUNCTION delete_pindex(index_name text)
RETURNS void AS $$
if index_name == '':
   raise plpy.Error('Pindex is not specified')
   raise Exception('Pindex is not specified')
from tspdb.src.pindex.pindex_managment import  delete_pindex
from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
# Build index 

delete_pindex(plpyimp(plpy),index_name, 'tspdb')
$$ LANGUAGE plpythonu;


CREATE or REPLACE FUNCTION list_pindices(OUT index_name text, OUT value_columns text[], OUT relation text, OUT time_column text, OUT initial_index bigint, OUT last_index bigint, OUT initial_timestamp timestamp,OUT last_timestamp timestamp, OUT agg_interval double precision,out uncertainty_quantification boolean)
RETURNS setof record LANGUAGE SQL AS $$
select b.index_name as index_name, array_agg(a.value_column) as value_column, b.relation, time_column, initial_index,  last_index ,initial_timestamp,  last_timestamp, agg_interval, uq as uncertainty_quantification
from  tspdb.pindices_columns as a 
JOIN tspdb.pindices as b 
on a.index_name = b.index_name 
GROUP BY b.index_name 
$$;
 

CREATE or REPLACE FUNCTION test_tspdb()
RETURNS void LANGUAGE plpythonu AS $$
from tspdb.src.database_module.plpy_imp import plpyimp
from tspdb.tests.test_module import create_tables, create_pindex_test
plpy.notice('Libraries Imported')
interface = plpyimp(plpy)
create_tables(interface)
plpy.notice('Sample Tables created .. Creating Pindices')
create_pindex_test(interface,'mixturets_var', 10000,100000, 2, 1, True, index_name = 'mixturets_var_pindex', time_column = 'time',agg_interval = 1. )
create_pindex_test(interface,'mixturets2', 2000000,2000000, 3, 1, True, index_name = 'mixturets2_pindex', time_column = 'time',agg_interval = 1. )
create_pindex_test(interface,'ts_basic_ts_5_5', 100000,1000000, 2, 1, True, index_name = 'basic_pindex', time_column = 'time',agg_interval = 5 )
plpy.notice('Pindices successfully created')
$$;
