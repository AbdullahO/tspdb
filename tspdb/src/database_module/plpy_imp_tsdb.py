import numpy as np
import pandas as pd
from tspdb.src.database_module.db_class import Interface
from tspdb.src.database_module.plpy_imp import plpyimp
#######################TO DO##########################
#1 get SUV instead of all getU,getS, getV
######################################################
class plpyimp(plpyimp):
    

    def get_time_series(self, name, start, end = None, start_ts = '1970/01/01 00:00:00', value_column="ts", index_column='row_id', Desc=False, interval = 60, aggregation_method = 'average' ):

        """
        query time series table to return equally-spaced time series values from a certain range  [start to end]
        or all values with time stamp/index greater than start  (if end is None)
        ----------
        Parameters
        ----------
        name: string 
            table (time series) name in database

        start: int or  timestamp
            start index (timestamp) of the range query

        end: int, timestamp 
            last index (timestamp) of the range query

        value_column: string
            name of column than contain time series value

        index_column: string  
            name of column that contains time series index/timestamp

        interval: float optional (default=60) 
            if time-index type is timestamp, determine the period (in seconds) in which the timestamps are truncated  

        aggregation_method: str optional (default='average') 
            the method used to aggragte values belonging to the same interval. options are: 'average', 'max', 'min', and 'median'  
        
        desc: boolean optional (default=false) 
            if true(false),  the returned values are sorted descendingly (ascendingly) according to index_column 
        ----------
        Returns
        ----------
        array, shape [(end - start +1) or  ceil(end(in seconds) - start(in seconds) +1) / interval ]
            Values of time series in the time interval start to end sorted according to index_col
        """
        # check if hypertable

        hypertable = self.engine.execute("SELECT count(*)>=1 as h FROM timescaledb_information.hypertable WHERE table_schema='public' AND table_name='%s';" % name)[0]['h']
            
        if isinstance(start, (int, np.integer)) and (isinstance(end, (int, np.integer)) or end is None):
            if end is None:
                sql = 'Select ' + value_column + " from  " + name + " where " + index_column + " >= "+str(start)+" order by "+index_column
                result = self.engine.execute(sql)

            else:
                if not Desc:
                    sql = 'Select ' + value_column + " from  " + name + " where " + index_column + " >= "+str(start)+" and " + index_column + " <= "+str(end)+" order by " + index_col
                else:
                    sql = 'Select ' + value_column + " from  " + name + " where " + index_column + " >= "+str(start)+" and " + index_column + " <= "+str(end)+" order by " + index_col + ' Desc'
                result = self.engine.execute(sql)
            result = [row for row in result]
            columns = value_column.split(',')
            return [[row[ci] for ci in columns] for row in result]

        elif  isinstance(start, (pd.Timestamp)) and (isinstance(end, (pd.Timestamp)) or end is None):
            #SELECT
            #time_bucket_gapfill('00:00:05', time) AS date,
            #avg(ts)
            #FROM ts_basic_ts_5_5
            #WHERE time >= '1/10/2012' AND time < ' 2012-10-06 18:53:15'
            #GROUP BY date
            #ORDER BY date;

            
            seconds = interval%60
            minutes = int(interval/60)
            hours = int(interval/3600)
            interval_str = '%s:%s:%s'%(hours, minutes, seconds)
            
            agg_func_dict = {'average': 'AVG', 'min': 'MIN', 'max': 'MAX'}
            try:
                agg_function = agg_func_dict[aggregation_method]
            except KeyError as e:
                print ('aggregation_method not valid choose from ("average", "min", "max"), Exception: "%s"' % str(e))
                raise
            ## might be needed
            start_ts_str = start_ts.strftime('%Y-%m-%d %H:%M:%S')
            ## fix strings formatting
            if hypertable:
                if end is None:
                    sql = "SELECT time_bucket_gapfill('%s', "+index_column+") AS date, "+agg_function+"("+value_column+") as avg_val FROM "+name+" where "+index_column+" >= '%s' and  "+index_column+" <= '%s' GROUP BY date ORDER BY date"
                    sql = sql%(interval_str, start.strftime('%Y-%m-%d %H:%M:%S'), 'now()',)
                else:
                    sql = "SELECT time_bucket_gapfill('%s', "+index_column+") AS date, "+agg_function+"("+value_column+") as avg_val FROM "+name+" where "+index_column+" >= '%s' and  "+index_column+" <= '%s' GROUP BY date ORDER BY date"
                    sql = sql%(interval_str, start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'),)
            else:
                if end is None:
                    select_sql = "select "+agg_function+"(m."+value_column+") avg_val from "+name+" m right join intervals f on m."+index_column+" >= f.start_time and m."+index_column+" < f.end_time where f.end_time > "+start_ts_str+"  group by f.start_time, f.end_time order by f.start_time"
                    generate_series_sql = "with intervals as (select n as start_time,n+'"+interval_str+"'::interval as end_time from generate_series('"+start.strftime('%Y-%m-%d %H:%M:%S')+"'::timestamp, now(),'"+interval_str+"'::interval) as n )"
                else:
                    generate_series_sql = "with intervals as (select n as start_time,n+'"+interval_str+"'::interval as end_time from generate_series('%s'::timestamp, '%s'::timestamp,'"+interval_str+"'::interval) as n )" 
                    generate_series_sql = generate_series_sql % (start_ts_str,end.strftime('%Y-%m-%d %H:%M:%S'),)
                    select_sql = "select "+agg_function+"(m."+value_column+") avg_val from "+name+" m right join intervals f on m."+index_column+" >= f.start_time and m."+index_column+" < f.end_time where f.end_time > '%s' and  f.start_time < '%s' group by f.start_time, f.end_time order by f.start_time" 
                    select_sql = select_sql%(start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'),)
                
                
                sql = generate_series_sql+ select_sql
            if Desc: sql += 'DESC'
            result = self.engine.execute(sql)
            result = [row for row in result]
            return [(row['avg_val'],) for row in result]

        else:
             raise Exception('start and end values must either be integers or pd.timestamp')

     
