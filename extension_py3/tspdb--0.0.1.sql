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

CREATE TABLE IF NOT EXISTS tspdb.pindices_stats (
  index_name text PRIMARY key ,
  number_of_observations bigint,
  number_of_trained_models bigint,
  imputation_score double precision,
  forecast_score double precision
);



CREATE or REPLACE FUNCTION create_pindex (table_name text, ts_column text, value_column text, index_name text, timescale boolean DEFAULT false, t_var int DEFAULT -1 ,k int DEFAULT 3 , k_var int DEFAULT 1, "T" int DEFAULT 2500000, "T0" int DEFAULT 1000,var_direct boolean DEFAULT true,gamma numeric DEFAULT 0.5, col_to_row_ratio int DEFAULT 10, agg_interval numeric DEFAULT 1.0, "L" int DEFAULT 0 )
RETURNS void AS $$
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex
if timescale:
    from tspdb.src.database_module.plpy_imp_tsdb import plpyimp
else:
    from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
pass
# Build index 
time_series_table = [table_name,value_column,ts_column]
if L == 0:
  TSPD = TSPI(T = T,T_var = T, rank = k, rank_var =  k_var, gamma = gamma, col_to_row_ratio = col_to_row_ratio, interface= plpyimp(plpy) ,time_series_table = time_series_table, recreate = True, direct_var = var_direct, index_name = index_name, agg_interval = agg_interval)
else:
  TSPD = TSPI(T = T,T_var = T, rank = k, rank_var =  k_var, gamma = gamma, L = L, interface= plpyimp(plpy) ,time_series_table = time_series_table, recreate = True, direct_var = var_direct, index_name = index_name, agg_interval = agg_interval)
TSPD.create_index()


$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION predict (table_name text, value_column text, t int,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95, projected boolean DEFAULT false,  OUT prediction numeric, OUT LB numeric,OUT UB numeric)
AS $$

from tspdb.src.pindex.predict import get_prediction
from tspdb.src.database_module.plpy_imp import plpyimp
#check if index exist or if there exist index that is implemented for column 

# get 
index_name_ = 'tspdb.'+index_name
if not uq:
  prediction = get_prediction( index_name_, table_name, value_column, plpyimp(plpy), t, uq, projected = projected)
  return prediction, 0,0
else: 
  prediction,interval = get_prediction( index_name_, table_name, value_column,  plpyimp(plpy), t, uq, projected = projected, uq_method = uq_method, c = c)
  return prediction, prediction-interval, prediction+ interval
$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION predict (table_name text, value_column text, t timestamp,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,projected boolean DEFAULT false,  OUT prediction numeric, OUT LB numeric,OUT UB numeric)
AS $$

from tspdb.src.pindex.predict import get_prediction
from tspdb.src.database_module.plpy_imp import plpyimp
#check if index exist or if there exist index that is implemented for column 

# get 
index_name_ = 'tspdb.'+index_name
if not uq:
  prediction = get_prediction( index_name_, table_name, value_column, plpyimp(plpy), t, uq, projected = projected)
  return prediction, 0,0
else: 
  prediction,interval = get_prediction( index_name_, table_name, value_column, plpyimp(plpy), t, uq, projected = projected, uq_method = uq_method, c = c)
  return prediction, prediction-interval, prediction+ interval
$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION predict_range (table_name text, value_column text,  t1 int, t2 int,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95, projected boolean DEFAULT false,OUT prediction numeric, OUT LB numeric,OUT UB numeric)
RETURNS SETOF record AS $$
from tspdb.src.pindex.predict import get_prediction_range
from tspdb.src.database_module.plpy_imp import plpyimp

#check if index exist or if there exist index that is implemented for column 
index_name_ = 'tspdb.'+index_name
# get 
if not uq:
  prediction = get_prediction_range( index_name_, table_name, value_column, plpyimp(plpy), t1,t2, uq, projected = projected)
  return zip(prediction, prediction,prediction)
else: 
  prediction,interval = get_prediction_range( index_name_, table_name, value_column, plpyimp(plpy), t1,t2,uq, projected = projected, uq_method = uq_method, c = c)
  return zip(prediction, prediction-interval, prediction+ interval)
$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION predict_range (table_name text, value_column text,  t1 timestamp, t2 timestamp,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,projected boolean DEFAULT false, OUT prediction numeric, OUT LB numeric,OUT UB numeric)
RETURNS SETOF record AS $$
from tspdb.src.pindex.predict import get_prediction_range
from tspdb.src.database_module.plpy_imp import plpyimp

#check if index exist or if there exist index that is implemented for column 
index_name_ = 'tspdb.'+index_name
# get 
if not uq:
  prediction = get_prediction_range( index_name_, table_name, value_column, plpyimp(plpy), t1,t2, uq,projected = projected)
  return zip(prediction, prediction,prediction)
else: 
  prediction,interval = get_prediction_range( index_name_, table_name, value_column,  plpyimp(plpy), t1,t2,uq, uq_method = uq_method, c = c,projected = projected)
  return zip(prediction, prediction-interval, prediction+ interval)
$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION forecast_next (table_name text, value_column text, time_column text ,  index_name text, ahead int DEFAULT 1,  averaging text DEFAULT 'average', uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95)
RETURNS setof numeric AS $$
from tspdb.src.database_module.plpy_imp import plpyimp
from tspdb.src.pindex.predict import forecast_next

#check if index exist or if there exist index that is implemented for column 
index_name_ = 'tspdb.'+index_name
a = forecast_next(index_name_,table_name, value_column, time_column, plpyimp(plpy), ahead = ahead,  averaging = averaging)
return a
$$ LANGUAGE plpython3u;



CREATE or REPLACE FUNCTION update_pindex(index_name text)
RETURNS void AS $$
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex
from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
# Build index 
TSPD = load_pindex(plpyimp(plpy),index_name)
TSPD.update_index()
$$ LANGUAGE plpython3u;

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
$$ LANGUAGE plpython3u;


CREATE or REPLACE FUNCTION list_pindices(OUT index_name text, OUT value_columns text[], OUT relation text, OUT time_column text,  OUT initial_timestamp text,OUT last_timestamp text, OUT agg_interval text,out uncertainty_quantification boolean)
RETURNS setof record LANGUAGE SQL AS $$
select b.index_name as index_name, array_agg(a.value_column) as value_column, b.relation, time_column, cast(initial_index as text) as start_timestamp,  cast (last_index as text) as last_timestamp, cast(agg_interval as text) || ' units' as agg_interval, uq as uncertainty_quantification
from  tspdb.pindices_columns as a 
JOIN tspdb.pindices as b 
on a.index_name = b.index_name 
where initial_index is not null
GROUP BY b.index_name 
union 
select b.index_name as index_name, array_agg(a.value_column) as value_column, b.relation, time_column, cast( initial_timestamp as text) as start_timestamp,  cast(last_timestamp as text) as last_timestamp, cast(cast(cast(agg_interval as text) as interval) as text) as agg_interval, uq as uncertainty_quantification
from  tspdb.pindices_columns as a 
JOIN tspdb.pindices as b 
on a.index_name = b.index_name 
where initial_timestamp is not null
GROUP BY b.index_name 
$$;

CREATE or REPLACE FUNCTION test_tspdb()
RETURNS void LANGUAGE plpython3u AS $$
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



CREATE or REPLACE FUNCTION pindices_stat(OUT index_name text, OUT number_of_observations bigint, OUT number_of_trained_models bigint, OUT imputation_score double precision,  OUT forecast_score double precision)
RETURNS setof record LANGUAGE SQL AS $$
select * from tspdb.pindices_stats;
$$;


