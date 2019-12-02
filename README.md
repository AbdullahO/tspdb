# tspDB - Time Series Predict DB
Time Series Forecasting and Imputation implemedted on top of PostgreSQL

This work has the following dependencies:

- PostgreSQL 12+
- Python 3.5+ with the libraries: (numpy, pandas, scipy, sklearn)
 

### Installation (Mac OS)

#### Prerequisites (start from here, the later stages assume that you downloaded postgres this way):

1- Install postgreSQL 11.5 from here: https://www.enterprisedb.com/thank-you-downloading-postgresql?anid=1256715

2- After the installation is done, we need to download the Language Pack. To do that, run Stack Builder (which was installed with PostgreSQL11) and follow these steps: 
		a- From the first screen of stack builder, choose the postgres installation you want to install the language pack on: (you should select PostgreSQL 11.5, your only option if you haven't installed PostgreSQL before) and click next
		b- from the application list choose 'Add-ons, tools, and utilities -> EDB Language Pack '
		c- Proceed to download and install the language package.

3- Now we will make sure that PL\Python is working in postgres. To do so:
		a- for convenience sake, add the postgreSQL directory to the path:
				
	sudo vim /etc/paths
click 'i' to go into insert mode add '/Library/PostgreSQL/11/bin' to the first line then click 'Esc' and ':x' to save and quit 
restart the terminal
	
b- run  postgreSQL using the command:
				
	psql -U postgres

c- from inside postgres:
				
	CREATE EXTENSION plpython3u;

d- if you see  "CREATE EXTENSION" that means it is working properly.


#### Installing tspdb

1- Download the package through cloning the reopsitory 

	git clone https://github.com/AbdullahO/tspdb.git
	

2-  cd into the package directory
				
	cd tspdb

3- run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the follwoing command:		
		
	sudo /Library/edb/languagepack-11/Python-3.6/bin/pip install . 

else, find the directory to the appropriate pip and run pip install . 

4- run:
		
	cd extension_py3 && sudo make install

This step uses pg_config to find out where PostgreSQL stores its extension files, if you have another installation of Postgres, this might not work. If it does not, copy the .control and the .sql files into the share/extension directory of your PostgreSQL installation

5- run postgreSQL using the command:
		
	psql -U postgres

6- create tspdb extension by running
		
	CREATE EXTENSION tspdb;

#### Testing tspdb

1- test the package through running 

	SELECT test_tspdb();

if you get at the last line 
	
	NOTICE:  Pindices successfully created
then tspdb should be working properly

2- check the built pindices through
		
	SELECT * from list_pindices();

this shows you a list of the predictive indices built in the aforementioned test

## Getting Started
The main functionalities of tspDB is enabling predictive queries, which are enabled via creating a prediction index on your time series table.
The index is created via the function `create_pindex()`. which you can use as follow:

``` sql
select create_pindex('tablename','time_column_name','{"value_column_name"}','index_name')
``` 
To get you familiar with tspDB capabilities, we provided a testing function that will create a set of time series tables in youe database. The test function will also create several prediction indices. Run the function from your postgres terminal
```sql
SELECT test_tspdb();
```
if you get at the last line
```
NOTICE:  Pindices successfully created
```
then the test has passed. Now we can check the predictive indices the test has created through

```sql
SELECT * FROM list_pindices();
```

You will see the three predictive indices created by the test. Now let's create our own predictive index, which we will call 'pindex1' on the time series table `mixturets2`. The prediction index is created on the time column `time` and the value column `ts_7`:
```sql
SELECT create_pindex('mixturets2','time','{"ts_7"}','pindex1');
```
we can see our newly created index by running `list_pindices` again:
```sql
SELECT * FROM list_pindices();
```

Let's now use that prediction index to prodice some *predictions*! let's for example predict at a time t that exists in the database. Effectively, we are *denoising* the existing observation or *imputing* a null observation. For example, at time 1, `ts_7` has a null value as  you can see by running:
```sql
SELECT ts_7 FROM mixturets2 WHERE time = 1;
```
Let's *impute* this point by running:
```sql
SELECT * FROM predict('mixturets2','ts_7',10,'pindex1');
```
Which will return predictions as well as upper an lower bound for a 95% confidence interval. We can get a tighter bound with lower confidence by changing the confidence interval to, say 80%: 
```sql
SELECT * FROM predict('mixturets2','ts_7',10,'pindex1', c=> 80);
```
The prediction index also support forecasting queries using the same function. For example, you can forecast the value of column `ts_7` at time 100010, ten points ahead of what exists in the database by running:
```sql
SELECT * FROM predict('mixturets2','ts_7',100010,'pindex1');
```
In a similar fashion, you can execute range predictive queries using `predict_range()`. for example, we can *impute* the first hundered points of `ts_7` using:

```sql
SELECT * FROM predict_range('mixturets2','ts_7',0,100,'pindex1');
```

or *forecast* the next 10 points using:

```sql
SELECT * FROM predict_range('mixturets2','ts_7',100001,100010,'pindex1');
```
