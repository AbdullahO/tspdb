# ReadMe

Start the docker container with the following command:

```bash
docker-compose up
```

Test the connection to the database with the following command:

```py
import psycopg2

## fill your db information
database = 'postgres'
user = 'postgres'
host = 'localhost'
password = 'postgres'
conn_string = "host='%s' dbname='%s' user='%s' password='%s'" % (host, database, user, password)
conn = psycopg2.connect(conn_string)
```

You can access the postgres database with the following command:

```bash
docker exec -it <container id> psql -U postgres
```

# Testing TSPDB

You can follow this [link](https://tspdb.mit.edu/installation/) or do the following -

Let's test the plpython3 package. Run PostgreSQL using the command (specify the appropriate user name and database as needed):
```sql
CREATE EXTENSION plpython3u;
```

Test the package by running the following command in psql terminal:
```sql
SELECT test_tspdb();
```

if you get at the last line
```
NOTICE:  Pindices successfully created
```

then tspdb should be working properly
