
# Installation 
For installing tspDB from the source files, follow the instructions depending on your system. Note that for tspDB to work you need to install PostgreSQL and plpython language pack. Details for installing them are listed below as well. We strongly recommend installing the prerequisites as shown below to avoid complications.
## Mac OS
If you have downloaded tspDB before and you want to upgrade to the current version, go to [here](installation.md#upgrade-tspdb)
### Prerequisites (PostgreSQL and plpython)
**(start from here, the later stages assume that you downloaded Postgres this way):**

1. Install postgres.app (with PostgreSQL version 14) [here](https://postgresapp.com). It is preferred to follow the instructions therein, but we list them here for convenience. 
    -  Download postgres.app (with PostgreSQL version 14) from [here](https://postgresapp.com/downloads.html), move it to the application folder when prompted, and double click on it.
    - Click "Initialize" to create a new server.
    - Configure your `$PATH` to use the included command line tools:
    
    ```
    sudo mkdir -p /etc/paths.d &&
    ```
    
    ```
    echo /Applications/Postgres.app/Contents/Versions/latest/bin | sudo tee /etc/paths.d/postgresapp
    ```

    - Restart the terminal for the changes to take effect. 

2. After the installation is done, we need to install plpython. To do that, we need to install Python 3.9.x  (universal2) from [here](https://www.python.org/downloads/macos/). Note that Other Python installations or versions are not supported. 

    - Once Python is installed, make sure that the path `/Library/Frameworks/Python.framework/Versions/3.9/bin` exist, since this is where postgres.app is expecting the Python interpreter to be. 

	
    - Run  PostgreSQL using the command (specify the appropriate user name and database as needed):

         
         ```				
         psql -U postgres
         ``` 
    
    - From inside PostgreSQL:

         ```				
         CREATE EXTENSION plpython3u;
         ```

    - If you see  "CREATE EXTENSION" then the extension is working properly.


- For updated instructions, refer to the postgres.app [docs](https://postgresapp.com/documentation/plpython.html).


### Installing tspDB

1- Download the package by cloning the repository 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```

2-  Go into the package directory
```sh				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the following command:		
```sh		
/Library/Frameworks/Python.framework/Versions/3.9/bin/pip3 install .
```
else, find the directory to the appropriate pip and run `pip install .` 

4- Run:
```sh		
cd extension_py3 && sudo make install
```
This step uses `pg_config` to find out where PostgreSQL stores its extension files, if you have another installation of Postgres, this might not work. If it does not, copy the .control and the .sql files into the share/extension directory of your PostgreSQL installation. 

5- run PostgreSQL using the command (specify the appropriate user name and database as needed):
``` sh		
psql -U postgres
```
6- create tspdb extension by running
``` SQL		
CREATE EXTENSION tspdb;
```
**Note:** if you get the error that the tspdb.control file does not exist, then step four did not run as it should. Thus do this command from inside the extension_py3 folder instead:
```sh
sudo cp -r . /Library/PostgreSQL/14/share/postgresql/extension/
```
and try to create the extension again. 

**Note:** the create extension command creates the extension in the database only. You need to run that command on each database you want to use tspdb on.

Finally, test your installation [here](installation.md#testing-tspdb) to make sure everything is working properly.

### Upgrading tspDB

**Caution**: upgrading tspdb will remove the pindices you currently have.

1- Download the package by cloning the  repository 
```sh
git clone https://github.com/AbdullahO/tspdb.git
```
2-  Go into the package directory
```sh				
cd tspdb
```
3- Run pip for the python Postgres uses, if you downloaded Postgres the same way described above, use the following command:		
```sh		
/Library/Frameworks/Python.framework/Versions/3.9/bin/pip3 install . --upgrade 
```
else, find the directory to the appropriate pip and run `pip install . --upgrade` 

4- Run:
```sh		
cd extension_py3 && sudo make install
```

5- run PostgreSQL using the command (specify the appropriate user name and database as needed):
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
**(start from here, the later stages assume that you downloaded Postgres this way):**

1- Install PostgreSQL 14 along with plpython via the following steps:

- Add PostgreSQL packages to your Ubuntu repository via 

```bash	
sudo apt-get install wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
sudo apt-get update
```

- Install PostgreSQL 14 with plpython3 using the following command
```sh
sudo apt-get install postgresql-plpython3-14
```	
2- Now let's test the plpython3 package.  Run  PostgreSQL using the command (specify the appropriate user name and database as needed):
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
sudo curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py
sudo apt-get install python3-distutils
sudo python3.6 get-pip.py 
hash -d pip3
```

### Installing tspdb

1- Download the package by cloning the repository, make sure you know where the path to the package is 
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
sudo cp tspdb.control /usr/share/postgresql/14/extension/tspdb.control
sudo cp tspdb--0.0.1.sql /usr/share/postgresql/14/extension/tspdb--0.0.1.sql
```
5- Run PostgreSQL using the command (specify the appropriate user name and database as needed):
```sh		
sudo -u postgres psql postgres
```
6- Create tspdb extension by running
```sql	
CREATE EXTENSION tspdb;
```
**Note:** the create extension command creates the extension in the database only. You need to run that command on each database you want to use tspdb on.

Finally, test your installation [here](installation.md#testing-tspdb) to make sure everything is working properly.

### Upgrading tspDB

**Caution**: upgrading tspdb will remove the pindices you currently have.

1- Download the package by cloning the repository 
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
sudo cp tspdb.control /usr/share/postgresql/14/extension/tspdb.control
sudo cp tspdb--0.0.1.sql /usr/share/postgresql/14/extension/tspdb--0.0.1.sql
```
5- Run PostgreSQL using the command (specify the appropriate user name and database as needed):
```sh		
sudo -u postgres psql postgres
```

6- Drop and create tspdb extension by running
``` SQL		
DROP EXTENSION tspdb CASCADE;
CREATE EXTENSION tspdb;
```


## Testing tspdb

1- Test the package by running the following command in psql terminal
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
this shows you a list of the predictive indices built in the aforementioned test. You should see three prediction indices built. 

With that, you are all set to test tspDB functionalities. Start from [here](index.md#getting-started) to get started
Project presentations logistics and final report submission