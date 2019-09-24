# Installation 

## Prerequisite (start from here, the later stages assume that you downloaded postgres this way):

1- Install postgreSQL 11.5 from here: https://www.enterprisedb.com/thank-you-downloading-postgresql?anid=1256715

2- After the installation is done, we need to download the Language Pack. To do that, run Stack Builder (which was installed with PostgreSQL11) and follow these steps: 
		a- From the first screen of stack builder, choose the postgres installation you want to install the language pack on: (you should select PostgreSQL 11.5, your only option if you haven't installed PostgreSQL before) and click next
		b- from the application list choose 'Add-ons, tools, and utilities -> EDB Language Pack '
		c- Proceed to download and install the language package.

3- Now we will make sure that PL\Python is working in postgres. To do so:
		a- for convenience sake, add the postgreSQL directory to the path:
				sudo vim /etc/paths
						click 'i' to go into insert mode
						add '/Library/PostgreSQL/11/bin' to the first line
						click 'Esc' and ':x' to save and quit 
				re-run the terminal
		b- ruSn  postgreSQL using the command:
				psql -U postgres
		c- from inside postgres:
				CREATE EXTENSION plpython3u
		d- if you see  "CREATE EXTENSION" that means it is working properly.


## Installing tspdb

1- Download through: .. (I'll send it in an email for now)

2- Extract the package and from the terminal cd into the package directory

3- run:		
		sudo /Library/edb/languagepack-11/Python-3.6/bin/pip install . 

4- run:
		cd extension_py3
		& 
		sudo make install

5- run postgreSQL using the command:
		psql -U postgres

6- create tspdb extension by running
		CREATE EXTENSION tspdb

## Testing tspdb

1- test the package through running 
		SELECT test_tspdb();
	if you get at the last line 
	NOTICE:  Pindices successfully created
	then tspdb should be working properly

2- check the built pindices through
		SELECT * from list_pindices()
	this shows you a list of the predictive indices built in the aforementioned test

## Example use cases
..... To be updated later .....

1- try to create another pindex: 
				
				select create_pindex('mixturets2','time','ts_7','pindex1');
2- list 
				select * from list_pindices();

3- insert and list

				\d mixturets2
				insert into mixturets2 values (100001, 27.0,27.0,27.0,27.0,1);
				select * from list_pindices();

4- predict with confidence interval and two methods
				
				select * from predict('mixturets2','ts_7',10,'pindex1');
				select * from predict('mixturets2','ts_7',10,'pindex1', uq_method=> 'Chebyshev');
				select * from predict('mixturets2','ts_7',10,'pindex1', uq_method=> 'Chebyshev', c => 50);

5- predict range
				select * from predic_range('mixturets2','ts_7',10,50,'pindex1', c => 90);

6- to predict and compare prediction and means:

				select time, ts_7, prediction, means from mixturets2  left join  predict('mixturets2','ts',time::int,'pindex1') on true where time > 1000 and time < 1100;

