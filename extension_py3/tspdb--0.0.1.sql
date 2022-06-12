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
  index_name text,
  column_name text,
  number_of_observations bigint,
  number_of_trained_models bigint,
  imputation_score double precision,
  forecast_score double precision,
  test_forecast_score double precision,
   PRIMARY KEY (index_name, column_name)
);





CREATE or REPLACE FUNCTION create_pindex (table_name text, time_column text, value_column text[], index_name text, fill_in_missing boolean DEFAULT true,"normalize" boolean DEFAULT true,auto_update boolean DEFAULT true, timescale boolean DEFAULT false, t_var int DEFAULT -1 ,k int DEFAULT Null , k_var int DEFAULT 1, t int DEFAULT 2500000, t0 int DEFAULT 1000,var_direct boolean DEFAULT true,gamma numeric DEFAULT 0.5, col_to_row_ratio int DEFAULT 10, agg_interval numeric DEFAULT NULL, l int DEFAULT 0 )
RETURNS void AS $$
from tspdb.src.pindex.predict import get_prediction_range, get_prediction
from tspdb.src.pindex.pindex_managment import TSPI

if timescale:
    from tspdb.src.database_module.plpy_imp_tsdb import plpyimp
else:
    from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 

# Build index 
L, T0 = l, t0
if L == 0:
  TSPD = TSPI(T = t,T_var = t, rank = k, rank_var =  k_var, gamma = gamma, col_to_row_ratio = col_to_row_ratio, interface= plpyimp(plpy) ,time_column = time_column, value_column = value_column, time_series_table_name = table_name, recreate = True, direct_var = var_direct, index_name = index_name, agg_interval = agg_interval, normalize = normalize, auto_update = auto_update, fill_in_missing = fill_in_missing)
else:
  TSPD = TSPI(T = t,T_var = t, rank = k, rank_var =  k_var, gamma = gamma, L = L, interface= plpyimp(plpy) ,time_column = time_column, value_column =value_column , time_series_table_name = table_name, recreate = True, direct_var = var_direct, index_name = index_name, agg_interval = agg_interval, normalize = normalize, auto_update = auto_update, fill_in_missing = fill_in_missing)
plpy.notice('createing pindex: T = %s, L =%s'%(TSPD.ts_model.T,TSPD.ts_model.L,))
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

CREATE or REPLACE FUNCTION predict (table_name text, value_column text, t text,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,projected boolean DEFAULT false,  OUT prediction numeric, OUT LB numeric,OUT UB numeric)
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

CREATE or REPLACE FUNCTION predict (table_name text, value_column text,  t1 int, t2 int,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95, projected boolean DEFAULT false,OUT prediction numeric, OUT LB numeric,OUT UB numeric)
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

CREATE or REPLACE FUNCTION predict (table_name text, value_column text,  t1 text, t2 text,  index_name text, uq boolean DEFAULT true, uq_method text DEFAULT 'Gaussian', c double precision DEFAULT 95,projected boolean DEFAULT false, OUT prediction numeric, OUT LB numeric,OUT UB numeric)
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
from tspdb.src.pindex.pindex_managment import TSPI, load_pindex_u
from tspdb.src.database_module.plpy_imp import plpyimp
#check if table is ts and columns are of appropriate type 
# Build index 
index_name_ = index_name
if 'tspdb.' not in index_name_[:6]: 
    index_name_ = 'tspdb.'+index_name_
TSPD = load_pindex_u(plpyimp(plpy),index_name_)
if  TSPD:
  TSPD.update_index()
$$ LANGUAGE plpython3u;

CREATE or REPLACE FUNCTION delete_pindex(index_name text)
RETURNS void AS $$
if index_name == '':
   raise plpy.Error('Pindex is not specified')
   raise Exception('Pindex is not specified')
from tspdb.src.pindex.pindex_managment import  delete_pindex
from tspdb.src.database_module.plpy_imp import plpyimp
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



CREATE or REPLACE FUNCTION pindices_stat(OUT index_name text,OUT column_name text, OUT number_of_observations bigint, OUT number_of_trained_models bigint, OUT imputation_score double precision,  OUT forecast_score double precision, OUT test_forecast_score double precision)
RETURNS setof record LANGUAGE SQL AS $$
select * from tspdb.pindices_stats;
$$;


CREATE or REPLACE FUNCTION get_lowerbound(table_name text, value_column text, time_column text, number_of_observations int DEFAULT 1000, samples int DEFAULT 100, k int DEFAULT 3, discretization_method text DEFAULT 'quantization')
RETURNS  numeric LANGUAGE plpython3u AS $$
import pandas as pd
import numpy as np
from tspdb.src.tslb.tslb import get_lower_bound
sql_query = "Select " + value_column+ " from " +table_name+ " order by "+ time_column +" Desc limit %s"%number_of_observations
result = plpy.execute(sql_query)
result = [row for row in result][::-1]
df = pd.DataFrame(result)
nan_sum = np.sum(pd.isna(df[value_column]))
if nan_sum>0:
  plpy.notice('THe lower bound cannot be estimated if there are missing values. The column you selected has %s NaNs in the last %s observations'%(nan_sum,number_of_observations))
  return np.nan
lb = get_lower_bound(df[value_column], samples=samples, k=k, discretization_method=discretization_method)
return lb
$$;

