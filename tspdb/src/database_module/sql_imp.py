from tspdb.src.database_module.db_class import Interface
import psycopg2
from sqlalchemy import create_engine
import numpy as np
import io
from time import time
import pandas as pd
from sqlalchemy.types import *

class SqlImplementation(Interface):
    def __init__(self, driver="postgresql", host="localhost", database="querytime_test", user="aalomar",
                 password="AAmit32lids"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.engine = create_engine(driver + '://' + user + ':' + password + '@' + host + '/' + database)


    def get_time_series(self, name, start, end, start_ts = '1970/01/01 00:00:00', connection = None, value_column="ts", index_column='"rowID"', Desc=False, interval = 60, aggregation_method = 'average'):

        """
        query time series table to return equally-spaced time series values from a certain range  [start to end]
        or all values with time stamp/index greter than start  (if end is None)
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
            the method used to aggragte values belonging to the same interval. options are: 'average', 'max', and 'min'  
        
        desc: boolean optional (default=false) 
            if true(false),  the returned values are sorted descendingly (ascendingly) according to index_column 
        ----------
        Returns
        ----------
        array, shape [(end - start +1) or  ceil(end(in seconds) - start(in seconds) +1) / interval ]
            Values of time series in the time interval start to end sorted according to index_column
        """
        name = '"'+name+'"'
        index_column = '"'+index_column+'"'
        value_columns = value_column.split(',')
        value_columns = ['"'+i+'"' for i in value_columns]
        if connection is None:
            connection = self.engine.connect()
        if isinstance(start, (int, np.integer)) and (isinstance(end, (int, np.integer)) or end is None):

            if end is None:
                sql = 'Select ' + ','.join(value_columns) + " from  " + name + " where " + index_column + " >= %s order by "+index_column
                result = connection.execute(sql, (start)).fetchall()

            else:
                if not Desc:
                    sql = 'Select ' + ','.join(value_columns) + " from  " + name + " where " + index_column + " >= %s and " + index_column + " <= %s order by " + index_column
                else:
                    sql = 'Select ' + ','.join(value_columns) + " from  " + name + " where " + index_column + " >= %s and " + index_column + " <= %s order by " + index_column + ' Desc'
                result = connection.execute(sql, (start, end)).fetchall()
        
        elif  isinstance(start, (pd.Timestamp)) and (isinstance(end, (pd.Timestamp)) or end is None):
        
            seconds = interval%60
            minutes = int((interval%3600)/60)
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
            ## queried columns
            queried_columns = ','.join([agg_function+"(m."+value+') "ag_'+value[1:-1]+'"' for value in value_columns])
            ## fix strings formatting
            if end is None:
                select_sql = "select "+queried_columns + " from "+name+" m right join intervals f on m."+index_column+" >= f.start_time and m."+index_column+" < f.end_time where f.end_time > %s  group by f.start_time, f.end_time order by f.start_time"
                generate_series_sql = "with intervals as (select n as start_time,n+'"+interval_str+"'::interval as end_time from generate_series(%s::timestamp, now(),'"+interval_str+"'::interval) as n )"
                sql = generate_series_sql+ select_sql
                result = connection.execute(sql, (start_ts_str,start.strftime('%Y-%m-%d %H:%M:%S'),)).fetchall()
            else:
                generate_series_sql = "with intervals as (select n as start_time,n+'"+interval_str+"'::interval as end_time from generate_series(%s::timestamp, %s::timestamp,'"+interval_str+"'::interval) as n )"
                select_sql = "select "+queried_columns+ " from "+name+" m right join intervals f on m."+index_column+" >= f.start_time and m."+index_column+" < f.end_time where f.end_time > %s and  f.start_time <= %s group by f.start_time, f.end_time order by f.start_time"
                if Desc: select_sql += 'DESC'
                sql = generate_series_sql+ select_sql
                result = connection.execute(sql, (start_ts_str,end.strftime('%Y-%m-%d %H:%M:%S'), start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'),)).fetchall()
        else:
             raise Exception('start and end values must either be integers or pd.timestamp')


        return result

    def get_coeff_model(self, index_name, model_no):
        """
        query the c table to get the coefficients of the (model_no) sub-model 
        ----------
        Parameters
        ----------
        index_name: string
            pindex_name 
        

        models_no:int
            submodel for which we want the coefficients
        ----------
        Returns
        ---------- 
        array 
        queried values for the selected range
        """
        query = "SELECT coeffvalue FROM " + index_name + " WHERE modelno = %s   order by coeffpos Desc; "
        result = self.engine.execute(query,(model_no,)).fetchall()
        return np.array(result)
  
    def get_U_row(self, table_name, tsrow_range, models_range,k, return_modelno = False, return_weights_decom= False):

        """
        query the U matrix from the database tables  created via the prediction index. the query depend on the ts_row
        range [tsrow_range[0] to tsrow_range[1]] and model range [models_range[0] to models_range[1]] (both inclusive)
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        tsrow_range:list of length 2 
            start and end index  of the range query predicate on ts_row
        
        models_range:list of length 2 
            start and end index  of the range query predicate on model_no

        k: int
            number of singular values retained in the prediction index

        return_modelno: boolean optional (default=false) 
            if true,  submodel numbers are returned in the first column   
        ----------
        Returns
        ---------- 
        array 
        
        queried values for the selected range
        
        """
        
        columns = 'u' + ',u'.join([str(i) for i in range(1, k + 1)])

        if return_modelno :
            columns = 'modelno, '+columns
        if return_weights_decom:
            columns = columns + ',uw'+ ',uw'.join([str(i) for i in range(1, k + 1)])
        
        query = "SELECT "+ columns +" FROM " + table_name + " WHERE tsrow >= %s and tsrow <= %s and (modelno >= %s and modelno <= %s)  order by row_id; "
        result = self.engine.execute(query,
                                     (tsrow_range[0], tsrow_range[1], models_range[0], models_range[1],)).fetchall()
        return np.array(result)
  

    def get_V_row(self, table_name, tscol_range,k, value_index, models_range = [0,10**10], return_modelno = False, return_weights_decom = False):


        """
        query the V matrix from the database tables  created via the prediction index. the query depend on the ts_col
        range [tscol_range[0] to tscol_range[1]]  (inclusive)
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        tscol_range:list of length 2 
            start and end index  of the range query predicate on ts_col
        k: int
            number of singular values retained in the prediction index

        return_modelno: boolean optional (default=false) 
            if true,  submodel numbers are returned in the first column   
        ----------
        Returns
        ---------- 
        array 
        queried values for the selected range
        """
        
        
        columns = 'v'+ ',v'.join([str(i) for i in range(1, k + 1)])

        if value_index is None:
            times_series_predicate = ''
        else:
            times_series_predicate = 'time_series = %s and'%value_index
        if return_modelno :
            columns = 'modelno, '+columns
        if return_weights_decom:
            columns = columns + ',vw'+ ',vw'.join([str(i) for i in range(1, k + 1)])
        
        if models_range is None:
            query = "SELECT "+ columns +" FROM " + table_name + " WHERE "+times_series_predicate+" tscolumn >= %s and tscolumn <= %s  order by row_id; "
            result = self.engine.execute(query, ( tscol_range[0], tscol_range[1],)).fetchall()
        else:
            query = "SELECT " + columns + " FROM " + table_name + " WHERE "+times_series_predicate+" tscolumn >= %s and tscolumn <= %s and (modelno >= %s and modelno <= %s)   order by row_id; "
            result = self.engine.execute(query, ( tscol_range[0], tscol_range[1], models_range[0], models_range[1],)).fetchall()
        return np.array(result)

    def get_S_row(self, table_name, models_range, k ,return_modelno = False, return_weights_decom = False):
        """
        query the S matrix from the database tables  created via the prediction index. the query depend on the model
        range [models_range[0] to models_range[1]] ( inclusive)
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        models_range: list of length 2 
            start and end index  of the range query predicate on model_no
        
        k: int
            number of singular values retained in the prediction index

        return_modelno: boolean optional (default=false) 
            if true,  submodel numbers are returned in the first column   
        ----------
        Returns
        ---------- 
        array 
        queried values for the selected range
        """
        
        columns  = 's' + ',s'.join([str(i) for i in range(1, k + 1)])

        if return_modelno :
             columns = 'modelno, '+columns
        if return_weights_decom:
            columns = columns + ',sw'+ ',sw'.join([str(i) for i in range(1, k + 1)])
        
        query = "SELECT "+ columns +" FROM " + table_name + " WHERE modelno >= %s and modelno <= %s  order by modelno;"
        result = self.engine.execute(query, (models_range[0], models_range[1],)).fetchall()
        return np.array(result)

    def get_SUV(self, table_name, tscol_range, tsrow_range, models_range, k ,value_index , return_modelno = False):

        """
        query the S, U, V matric from the database tables created via the prediction index. the query depend on the model
        range, ts_col range, and ts_row range (inclusive ranges)
            
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        tscol_range:list of length 2 
            start and end index  of the range query predicate on ts_col
        
        tsrow_range:list of length 2 
            start and end index  of the range query predicate on ts_row
        
        models_range: list of length 2 
            start and end index  of the range query predicate on model_no
        
        k: int
            number of singular values retained in the prediction index

        return_modelno: boolean optional (default=false) 
            if true,  submodel numbers are returned in the first column   
        ----------
        Returns
        ---------- 
        S array 
        queried values for the selected range of S table

        U array 
        queried values for the selected range of U table

        V array 
        queried values for the selected range of V table

        """
        
        self.vcol = 'v'+ ',v'.join([str(i) for i in range(1, k + 1)])
        self.ucol = 'u' + ',u'.join([str(i) for i in range(1, k + 1)])
        self.scol = 's' + ',s'.join([str(i) for i in range(1, k + 1)])

        columns = self.scol
        if return_modelno :
                columns = 'modelno, '+columns
        query = "SELECT "+ columns +" FROM " + table_name + "_s WHERE modelno = %s or modelno = %s  order by modelno; "
        S = self.engine.execute(query, (models_range[0], models_range[1],)).fetchall()
        
        
        columns = self.vcol
        
        query = "SELECT " + columns + " FROM " + table_name + "_v WHERE tscolumn = %s and time_series = %s order by row_id; "
        V = self.engine.execute(query, (tscol_range[0],value_index)).fetchall()
        
        columns = self.ucol
        query = "SELECT "+ columns +" FROM " + table_name + "_u WHERE tsrow = %s and (modelno = %s or modelno = %s)  order by row_id; "
        U = self.engine.execute(query,
                                     (tsrow_range[0], models_range[0], models_range[1],)).fetchall()
        U,S,V = map(np.array, [U,S,V])
        
        return U,S,V
        

    def get_coeff(self, table_name, column):

        """
        query the LR coefficients from the database materialized view  created via the index. 
        the query need to determine the queried column
            
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        column: string optioanl (default = 'average' )
            column name, for possible options, refer to ... 
        
        ----------
        Returns
        ---------- 
        coeffs array 
            queried coefficients for the selected average
        """
        
        query = "SELECT %s from %s order by %s Desc" %(column , table_name, 'coeffpos')
        result = self.engine.execute(query).fetchall()

        return result

    def query_table(self, table_name, columns_queried ,predicate= '' ):
        """
        query columns from table_name according to a predicate
            
        ----------
        Parameters
        ----------
        table_name: string
            table name in database
        
        columns_queried: list of strings
            list of queries columns e.g. ['age', 'salary']
            
        predicate: string optional (default = '')
            predicate written as string e.g.  'age < 1'
        ----------
        Returns
        ---------- 
        result array 
            queried tuples

        """
        
        columns = '"' + '","'.join(columns_queried) +'"'
        if predicate == '':
            query = "SELECT %s from %s ;" % (columns, table_name,)
            return self.engine.execute(query).fetchall()
        else:
            query = "SELECT %s from %s where %s;" % (columns, table_name, predicate)
            return self.engine.execute(query).fetchall()

    def create_table(self, table_name, df, primary_key=None, load_data=True,replace_if_exists = True , include_index=True,
                     index_label="row_id", type_dict = None):
        """
        Create table in the database with the same columns as the given pandas dataframe. Rows in the df will be written to
        the newly created table if load_data.
        ----------
        Parameters
        ----------
        table_name: string
            name of the table to be created
        
        df: Pandas dataframe
             Dataframe used to determine the schema of the table, as well as the data to be written in the new table (if load_data)
        
        primary_key: str, optional (default None)
            primary key of the table, should be one of the columns od the df
        
        load_data: boolean optioanl (default True) 
            if true, load data in df to the newly created table via bulk_inset()

        replace_if_exists: boolean optioanl (default False) 
            if true, drop the existing table of the same name (if exists).

        include_index: boolean optioanl (default True)
            if true, include the index column of the df, with its name being index_column_name
        
        index_label: string optional (default "index")
            name of the index column of the df in the newly created database table 

        """

        # drop table if exists:
        if replace_if_exists:
            self.drop_table(table_name)

        elif self.table_exists(table_name):
            raise ValueError('table with %s already exists in the database!' % table_name)
        # create table in database
        schema_tablesplit = table_name.split('.') 
        if len(schema_tablesplit)== 2:
            table_name_ = schema_tablesplit[1]
            schema = schema_tablesplit[0]
        else:
            schema = None
            table_name_ = table_name
        df.head(0).to_sql(table_name_, self.engine, index=include_index, index_label=index_label, schema = schema, dtype=type_dict)
        
        
        if load_data:
            self.bulk_insert(table_name, df, include_index=include_index, index_label=index_label)

        if primary_key is not None:
            query = "ALTER TABLE  %s ADD PRIMARY KEY (%s);" % ( table_name, primary_key)
            self.engine.execute(query)
        # load content



    def drop_table(self, table_name):
        """
        Drop table from  database
        ----------
        Parameters
        ----------
        table_name: string
            name of the table to be deleted
        """


        query = " DROP TABLE IF EXISTS " +table_name + " Cascade; "

        self.engine.execute(query)

    def create_index(self, table_name, column, index_name='', ):
        """
        Constructs an index on a specified column of the specified table
        ----------
        Parameters
        ----------
        table_name: string 
            the name of the table to be indexed
        
        column: string 
            the name of the column to be indexed on
        
        index_name: string optional (Default '' (DB default))  
            the name of the index
        """
        
        query = 'CREATE INDEX %s ON %s (%s);' % (index_name  ,table_name,column)
        self.engine.execute( query)

    def create_coefficients_average_table(self, table_name, created_table_name, averages,max_model, refresh = False ):
        """
        Create the matrilized view where the coefficient averages are calculated. 
        ----------
        Parameters
        ----------
        table_name:  string 
            the name of the coefficient tables

        created_table_name:  string 
            the name of the created matrilized view
        
        average_windows:  list 
            windows for averages to be calculated (e.g.: [10,20] calc. last ten and 20 models)
        
        max_model:  int 
            index of the latest submodel

        :param average_windows:  (list) windows for averages to be calculated (e.g.: [10,20] calc. last ten and 20 models)
        """
        if refresh:
            self.engine.execute('REFRESH MATERIALIZED VIEW '+ created_table_name)
            return
        s1 =  'SELECT coeffpos, avg(coeffvalue) as average,'
        s_a =  'avg(coeffvalue) FILTER (WHERE modelno <= %s and modelno > %s -%s) as Last%s'
        predicates = (',').join([s_a %(max_model,max_model,i,i) for i in averages])
        query = s1+ predicates + ' FROM %s group by coeffpos' %table_name
        self.create_table_from_query(created_table_name, query)

    def create_table_from_query(self, table_name, query, ):
        """
        Create a new table using the output of a certain query. This is equivalent to a materialized view in
        PostgreSQL and Oracle
        ----------
        Parameters
        ----------
        table_name:  string 
            the name of the table to be indexed
        query: string 
            query to create table from
        """
        query = 'CREATE MATERIALIZED VIEW %s AS '%table_name+ query
        self.engine.execute( query)
        pass

    def execute_query(self, query, args = ''):
        """
        function that simply passes queries to DB
        ----------
        Parameters
        ----------
        query: string
            query to be executed
        ----------
        Returns
        ----------
        list
            query output
        """
   
        results = self.engine.execute(query,args).fetchall()
        

        return results

    def insert(self, table_name, row, columns =None):
        """
        Insert a new full row in table_name
        ----------
        Parameters
        ----------
        table_name:  string 
            name of an existing table to insert the new row to
        row: list
            data to be inserted
        """


        row = ["'"+str(i)+"'" if (type(i) is str or type(i) == pd.Timestamp) else i for i in row ]
        row = ["NULL" if pd.isna(i)  else i for i in row ]
        row = [str(i) for i in row]
        if columns is not None: columns = '('+','.join(columns)+')'
        else: columns = ''
        query = 'insert into '+table_name+ columns+ ' values (' + ','.join(row)+');'
        self.engine.execute(query)

    def bulk_insert(self, table_name, df, include_index=True, index_label="row_id"):
        """
        Insert rows in pandas dataframe to table_name
        ----------
        Parameters
        ----------
        table_name: string 
            name of the table to which we will insert data
        
        df pandas dataframe 
            Dataframe containing the data to be added
        """
        # preprocess tuples and lists into postgres arrays
        # columns = df.select_dtypes(['object']).columns
        # for column in columns:
        #     if type(df[column][0]) == tuple or type(df[column][0]) == list:
        #         df[column] = df[column].astype(str).str.replace('(','{').str.replace(')','}')
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=include_index, index_label=index_label)
        output.seek(0)
        cur.copy_from(output, table_name, null="")
        conn.commit()

    def table_exists(self, table_name, schema='public'):
        """
        check if a table exists in a certain database and schema
        ----------
        Parameters
        ----------
        table_name: string 
            name of the table
        
        schema: string default ('public')

        """
        return self.engine.execute('SELECT EXISTS(SELECT *  FROM information_schema.tables WHERE table_name = %s AND table_schema = %s );', (table_name,schema ,)).fetchone()[0]
    
    def delete(self, table_name, predicate):
        """
        check if a table exists in a certain database and schema
        ----------
        Parameters
        ----------
        table_name: string 
            name of the table contating the row to be deleted
        
        predicate: string
            the condition to determine deleted rows

        """
        
        if predicate == '': query = "DELETE from %s ;" % ( table_name)

        else: query = "DELETE from %s where %s;" % ( table_name, predicate)
        return self.engine.execute(query)

    def create_insert_trigger(self, table_name, index_name):
        # function = '''CREATE or REPLACE FUNCTION %s_update_pindex_tg() RETURNS trigger  AS $$ \n \
        # try: plpy.execute("select update_pindex('%s');") \n \
        # except: plpy.notice('Index is not updated, insert is carried forward') \n
        # $$LANGUAGE plpython3u;'''
        function = '''CREATE or REPLACE FUNCTION %s_update_pindex_tg() RETURNS trigger  AS $$ \n \
        plpy.execute("select update_pindex('%s');")\
        $$LANGUAGE plpython3u;'''
        self.engine.execute(function %(index_name, index_name))
        query = "CREATE TRIGGER tspdb_update_pindex_tg_%s AFTER insert ON "%index_name[6:] + table_name + " FOR EACH STATEMENT EXECUTE PROCEDURE " +index_name+"_update_pindex_tg(); "
        self.engine.execute(query)


    def drop_trigger(self, table_name, index_name):
        query = "DROP TRIGGER if EXISTS tspdb_update_pindex_tg_%s on "%index_name[:] + table_name 
        self.engine.execute(query)

    def get_extreme_value(self, table_name, column_name, extreme = 'min'):
        column_name  = '"'+column_name+'"'
        table_name = '"'+table_name+'"'
        agg_func_dict = { 'min': 'MIN', 'max': 'MAX'}
        query = "SELECT  %s(%s) from  "+ table_name
        ext = self.engine.execute(query %(agg_func_dict[extreme], column_name)).fetchone()[0]
        if isinstance(ext, (int, np.integer)): return ext
        else: return str(ext)



    def get_time_diff(self, table_name, time_column, number_of_pts = 100):
        """
        return the median in the difference between first 100 consecutive time points
        ----------
        Parameters
        ----------
        table_name: string 
            name of the table contating the row to be deleted
        
        time_column: string
            tname of time column

        number_of_pts: int
            Number of point to estimate the differnce Default:100 
        """
        time_column_ = '"'+time_column+'"'
        table_name_ = '"'+table_name+'"'

        query = "SELECT %s from %s order by %s limit %s;" % (time_column_, table_name_,time_column_, number_of_pts)
        result =  self.engine.execute(query)
        result = [row[time_column]  for row in result]
        if  isinstance(result[0], (int, np.integer)): 
            return np.median(np.diff(result))
        else:
            timestamps_float  = [pd.Timestamp(i).timestamp() for i in result]
            return np.median(np.diff(timestamps_float))


        