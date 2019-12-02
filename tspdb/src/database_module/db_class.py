import abc

class Interface(object):
    __metaclass__ = abc.ABCMeta
    @property
    def schema(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_time_series(self, name, start, end, value_column, index_column, interval = 60, aggregation_method = 'average', desc = False):
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

        index_col: string  
            name of column that contains time series index/timestamp

        interval: float optional (default=60) 
            if time-index type is timestamp, determine the period (in seconds) in which the timestamps are truncated  

        aggregation_method: str optional (default='average') 
            the method used to aggragte values belonging to the same interval. options are: 'average', 'max', 'min', and 'median'  
        
        desc: boolean optional (default=false) 
            if true(false),  the returned values are sorted descendingly (ascendingly) according to index_col 
        ----------
        Returns
        ----------
        array, shape [(end - start +1) or  ceil(end(in seconds) - start(in seconds) +1) / interval ]
            Values of time series in the time interval start to end sorted according to index_col
        """

    @abc.abstractmethod
    def get_U_row(self, table_name, tsrow_range, models_range, k, return_modelno = False):
        """
        query the U matrix from the database tables created via the predictin index. the query depend on the ts_row
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
        pass
    
    @abc.abstractmethod
    
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
        pass

    @abc.abstractmethod
    def get_V_row(self, table_name, tscol_range, k, return_modelno ):

        """
        query the V matrix from the database table created via the predictin index. the query depend on the ts_col
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
        
        pass



    @abc.abstractmethod
    def get_S_row(self, table_name, models_range,k, return_modelno = False):
        """
        query the S matrix from the database table created via the predictin index. the query depend on the model
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
        
        pass

    @abc.abstractmethod
    def get_SUV(self, table_name, tscol_range, tsrow_range, models_range, k ,return_modelno):
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
        
        pass


    @abc.abstractmethod
    def get_coeff(self, table_name, column = 'average'):
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
        pass

    @abc.abstractmethod
    def query_table(self, table_name, columns_queried = [],predicate= '' ):
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
        pass


    @abc.abstractmethod
    def create_table(self, table_name,df, primary_key=None, load_data=True, replace_if_exists = False , include_index=True,
                     index_label="index"):
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
        pass

    def drop_table(self, table_name,):
        """
        Drop table from  database
        ----------
        Parameters
        ----------
        table_name: string
            name of the table to be deleted
        """
        pass

    @abc.abstractmethod
    def create_index(self, table_name, column, index_name='',):
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
        pass

    @abc.abstractmethod
    def create_table_from_query(self, table_name, query):
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
        pass

    @abc.abstractmethod
    def execute_query(self, query):
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
        array
            query output
        """
        pass

    @abc.abstractmethod
    def insert(self,table_name, row):
        """
        Insert a new row in table_name
        ----------
        Parameters
        ----------
        table_name:  string 
            name of an existing table to insert the new row to
        row: list
            data to be inserted
        """

    @abc.abstractmethod
    def create_coefficients_average_table(self, table_name, created_table_name, average_windows, max_model, refresh = False ):
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
    
        average_windows:  list
             windows for averages to be calculated (e.g.: [10,20] calc. last ten and 20 models)
        
        refresh: Boolean
            if true, refresh view
        """

        pass
    @abc.abstractmethod
    def bulk_insert(self, table_name, df):
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
        pass
    @abc.abstractmethod
    def table_exists(self, table_name, schema = 'public'):
        """
        check if a table exists in a certain database and schema
        ----------
        Parameters
        ----------
        table_name: string 
            name of the table
        
        schema: string default ('public')

        """
    @abc.abstractmethod
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

    @abc.abstractmethod
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
