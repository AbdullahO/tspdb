
# Installation 
For installing tspDB from the source files, follow the instructions depending on your system. Note that for tspDB to work you need to install PostgreSQL and plpython lanuage pack. Details for installing them are listed below as well. We strongly recommend installing the prerequisites as shown below to avoid complications.
## Mac OS
If you have downloaded tspDB before and you want to upgrade to the current version, go to [here](installation.md#upgrade-tspdb)
### Prerequisites (PostgreSQL and plpython)
**(start from here, the later stages assume that you downloaded postgres this way):**

1- Install postgreSQL 12 fro Mac OS X from [here](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)

2- After the installation is done, we need to download the Language Pack. To do that, run Stack Builder (which was installed with PostgreSQL) and follow these steps: 
		
- From the first screen of stack builder, choose the postgres installation you want to install the language pack on: (you should select PostgreSQL 12.x, your only option if you haven't installed PostgreSQL before) and click next

- from the application list choose 'Add-ons, tools, and utilities -> EDB Language Pack '

- Proceed to download and install the language package.

3- Now we will make sure that PL\Python is working in postgres. To do so:
	
- For convenience sake, add the postgreSQL directory to the path:
```sh				
sudo vim /etc/paths
```
click 'i' to go into insert mode add '/Library/PostgreSQL/12/bin' to the first line then click 'Esc' and ':x' to save and quit.
Restart the terminal.
	
- Run  postgreSQL using the command (specify the appropriate user name and database as needed):
```sh				
psql -U postgres
```
- From inside postgres:
```sql				
CREATE EXTENSION plpython3u;
```
3- If you see  "CREATE EXTENSION" then the extension is working properly.


### Installing tspDB

1- Download the package through cloning the reopsitory 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```

2-  Go into the package directory
```sh				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the following command:		
```sh		
sudo /Library/edb/languagepack/v1/Python-3.7/bin/pip3 install . 
```
else, find the directory to the appropriate pip and run `pip install .` 

4- Run:
```sh		
cd extension_py3 && sudo make install
```
This step uses pg_config to find out where PostgreSQL stores its extension files, if you have another installation of Postgres, this might not work. If it does not, copy the .control and the .sql files into the share/extension directory of your PostgreSQL installation. 

5- run postgreSQL using the command (specify the appropriate user name and database as needed):
``` sh		
psql -U postgres
```
6- create tspdb extension by running
``` SQL		
CREATE EXTENSION tspdb;
```
**Note:** if you get ther error that the tspdb.control file does not exist, then step four did not run as it should. Thus do this command from inside the extension_py3 folder instead:
```sh
sudo cp -r . /Library/PostgreSQL/12/share/postgresql/extension/
```
and try to create the extension again. 

**Note:** the create extension command create the extension in the database only. You need to run that command on each database you want to use tspdb on.

Finally, test your installation [here](installation.md#testing-tspdb) to make sure everything is working properly.

### Upgrading tspDB

**Caution**: upgrading tspdb will remove the pindices you currently have.

1- Download the package through cloning the reopsitory 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```
2-  Go into the package directory
```sh				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the following command:		
```sh		
sudo /Library/edb/languagepack/v1/Python-3.7/bin/pip3 install . --upgrade 
```
else, find the directory to the appropriate pip and run `pip install . --upgrade` 

4- Run:
```sh		
cd extension_py3 && sudo make install
```

5- run postgreSQL using the command (specify the appropriate user name and database as needed):
``` sh		
psql -U postgres
```
6- Drop and create tspdb extension by running
``` SQL		
DROP EXTENSION tspdb CASCADE;
CREATE EXTENSION tspdb;
```

## Windows
### Prerequisites (PostgreSQL and plpython)
**(start from here, the later stages assume that you downloaded postgres this way):**

1- Install postgreSQL 12 from [here](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)

2- After the installation is done, we need to download the Language Pack. To do that, run Stack Builder (which was installed with PostgreSQL) and follow these steps: 
		
- From the first screen of stack builder, choose the postgres installation you want to install the language pack on: (you should select PostgreSQL 12, your only option if you haven't installed PostgreSQL before) and click next

- From the application list choose 'Add-ons, tools, and utilities -> EDB Language Pack '

- Proceed to download and install the language package.

3- Now we will make sure that PL\Python is working in postgres. To do so:
	
- Run command prompt (as an administrator) and then add Postgres and python to your system path, and add the system varaible PYTHONHOME by running the commands:
```bash
setx PATH "c:\edb\languagepack\v1\Python-3.7;c:\Program Files\PostgreSQL\12\bin;c:\edb\languagepack\v1\Python-3.7\Scripts;%PATH%"
setx PYTHONHOME "c:\edb\languagepack\v1\Python-3.7"
COPY "c:\edb\languagepack\v1\Python-3.7\python37.dll" "c:\Windows\System32\python37.dll" 
```

- Restart command prompt, run again as an adminstrator and run  postgreSQL using the command (specify the appropriate user name and database as needed):
				
```bash
psql -U postgres
```
Note that it will ask for your password, which you have set up during the installation.  

- From inside postgres run :
```sql			
CREATE EXTENSION plpython3u;
```
4- If you see  "CREATE EXTENSION" then this means it is working properly.


### Installing tspdb

1- Download the package through cloning the reopsitory 
```bash
git clone https://github.com/AbdullahO/tspdb.git
```

2-  Go into the package directory
```bash				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres and plpython the same way described above, use the following command:		
```bash				
c:\edb\languagepack\v1\Python-3.7\Scripts\pip install . 
```			

else, find the directory to the appropriate pip and run `pip install .` 

4- Copy extension files to their appropriate postgres folder:
```bash					
robocopy /s extension_py3  "c:\Program Files\PostgreSQL\12\share\extension"
```

This step copy the .control and the .sql files into the share/extension directory of your PostgreSQL installation. Let's install the extension now through:

5- Run postgreSQL using the command (specify the appropriate user name and database as needed):
```bash		
psql -U postgres
```
6- Create tspdb extension by running
```sql		
CREATE EXTENSION tspdb;
```
**Note:** the create extension command create the extension in the database only. You need to run that command on each database you want to use tspdb on.


Finally, test your installation [here](installation.md#testing-tspdb) to make sure everything is working properly.


### Upgrading tspDB

**Caution**: upgrading tspdb will remove the pindices you currently have.

1- Download the package through cloning the reopsitory 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```
2-  Go into the package directory
```sh				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the following command:		
```sh		
c:\edb\languagepack\v1\Python-3.7\Scripts\pip install .  --upgrade 
```
else, find the directory to the appropriate pip and run `pip install . --upgrade` 

4- Copy extension files to their appropriate postgres folder:
```bash					
robocopy /s extension_py3  "c:\Program Files\PostgreSQL\12\share\extension"
```

5- run postgreSQL using the command (specify the appropriate user name and database as needed):
``` sh		
psql -U postgres
```
6- Drop and create tspdb extension by running
``` SQL		
DROP EXTENSION tspdb CASCADE;
CREATE EXTENSION tspdb;
```

## Ubuntu 
### Prerequisites (PostgreSQL and plpython)
**(start from here, the later stages assume that you downloaded postgres this way):**

1- Install postgreSQL 12 along with plpython via the following steps:

- Add PostgreSQL packages to to your Ubuntu repository via 

```bash	
sudo apt-get install wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
sudo apt-get update
```

- Install PostgreSQL 12 with plpython3 using the following command
```sh
sudo apt-get install postgresql postgresql-plpython3-12 
```	
2- Now let's test the plpython3 package.  Run  postgreSQL using the command (specify the appropriate user name and database as needed):
```sh		
sudo -u postgres psql postgres
```
- From inside postgres:
```sql				
CREATE EXTENSION plpython3u;
```
 If you see  "CREATE EXTENSION", then it is working properly. use `\q` to quit postgres.

3- install pip for the python version used by postgres
```sh		
cd /usr/lib/python3.6
sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo apt-get install python3-distutils
sudo python3.6 get-pip.py 
hash -d pip3
```

### Installing tspdb

1- Download the package through cloning the reopsitory, make sure you know where the path to the package is 
```
git clone https://github.com/AbdullahO/tspdb.git
```	

2-  Go to the directory where python is installed
```sh				
cd /usr/lib/python3.6
```
3- Run pip install as follow, where \path_to_package\ is the directory where you cloned the tspdb repo:
```sh
sudo pip3 install /path_to_package/.
```
for example, if your package is downloaded at the user `ubuntu` directory, use `sudo pip3 install /home/ubuntu/tspdb`

4- Go to the extension_py3  directory in the package and run:
```sh		
sudo cp tspdb.control /usr/share/postgresql/12/extension/tspdb.control
sudo cp tspdb--0.0.1.sql /usr/share/postgresql/12/extension/tspdb--0.0.1.sql
```
5- Run postgreSQL using the command (specify the appropriate user name and database as needed):
```sh		
sudo -u postgres psql postgres
```
6- Create tspdb extension by running
```sql	
CREATE EXTENSION tspdb;
```
**Note:** the create extension command create the extension in the database only. You need to run that command on each database you want to use tspdb on.

Finally, test your installation [here](installation.md#testing-tspdb) to make sure everything is working properly.

### Upgrading tspDB

**Caution**: upgrading tspdb will remove the pindices you currently have.

1- Download the package through cloning the reopsitory 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```
2-  Go to the directory where python is installed
```sh				
cd /usr/lib/python3.6
```
3- Run pip install as follow, where \path_to_package\ is the directory where you cloned the tspdb repo:
```sh
sudo pip3 install /path_to_package/. --upgrade
```
for example, if your package is downloaded at the user `ubuntu` directory, use `sudo pip3 install /home/ubuntu/tspdb`

4- Go to the extension_py3  directory in the package and run:
```sh		
sudo cp tspdb.control /usr/share/postgresql/12/extension/tspdb.control
sudo cp tspdb--0.0.1.sql /usr/share/postgresql/12/extension/tspdb--0.0.1.sql
```
5- Run postgreSQL using the command (specify the appropriate user name and database as needed):
```sh		
sudo -u postgres psql postgres
```

6- Drop and create tspdb extension by running
``` SQL		
DROP EXTENSION tspdb CASCADE;
CREATE EXTENSION tspdb;
```


## Testing tspdb

1- Test the package through running the following command in psql terminal
```sql	
SELECT test_tspdb();
```
if you get at the last line 
	
	NOTICE:  Pindices successfully created
then tspdb should be working properly

2- Check the built pindices through
```sql			
SELECT * from list_pindices();
```
this shows you a list of the predictive indices built in the aforementioned test. You should see three prediction indeices built. 

With that, you are all set to test tspDB functionalities. Start from [here](index.md#getting-started) to get started
